import polars as pl
import os
import msgpack
from datetime import datetime
from flask import Flask, render_template, request
from flask_compress import Compress

app = Flask(__name__)
Compress(app)

# --- CONFIG ---
DATA_DIR = 'data'
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

COMPS_TSV = os.path.join(DATA_DIR, 'WCA_export_competitions.tsv')
COUNTRIES_TSV = os.path.join(DATA_DIR, 'WCA_export_countries.tsv')
RESULTS_TSV = os.path.join(DATA_DIR, 'WCA_export_results.tsv')

# Using .mp as requested
COMPS_CACHE = os.path.join(CACHE_DIR, 'comps.mp')
COUNTRIES_CACHE = os.path.join(CACHE_DIR, 'countries.mp')
RESULTS_CACHE = os.path.join(CACHE_DIR, 'results.mp')

EVENT_NAMES = {
    "333": "3x3 Cube", "222": "2x2 Cube", "444": "4x4 Cube", "555": "5x5 Cube",
    "666": "6x6 Cube", "777": "7x7 Cube", "333oh": "3x3 One-Handed",
    "333bf": "3x3 Blindfolded", "333fm": "3x3 Fewest Moves", "clock": "Clock", 
    "minx": "Megaminx", "pyram": "Pyraminx", "skewb": "Skewb", "sq1": "Square-1",
    "444bf": "4x4 Blindfolded", "555bf": "5x5 Blindfolded", "333mbf": "3x3 Multi-Blind",
    "333ft": "3x3 With Feet", "333mbo": "3x3 Multi-Blind Oldstyle",
    "magic": "Rubik's Magic", "mmagic": "Master Magic"
}

def pre_cache_data():
    """Converts heavy TSVs to tiny aggregated binary files."""
    if not os.path.exists(COMPS_CACHE) and os.path.exists(COMPS_TSV):
        df = pl.read_csv(COMPS_TSV, separator='\t', quote_char=None, ignore_errors=True)
        df = df.select(['id', 'name', 'country_id', 'year', 'month', 'day'])
        with open(COMPS_CACHE, 'wb') as f:
            f.write(msgpack.packb(df.to_dicts()))

    if not os.path.exists(COUNTRIES_CACHE) and os.path.exists(COUNTRIES_TSV):
        df = pl.read_csv(COUNTRIES_TSV, separator='\t', quote_char=None, ignore_errors=True)
        df = df.with_columns(pl.col("iso2").str.to_lowercase())
        data = df.select(["id", "iso2"]).to_dicts()
        with open(COUNTRIES_CACHE, 'wb') as f:
            f.write(msgpack.packb(data))

    # Aggressive Thinning: Pre-summarize results per competition
    if not os.path.exists(RESULTS_CACHE) and os.path.exists(RESULTS_TSV):
        print("Creating lightweight results summary...")
        df = pl.read_csv(RESULTS_TSV, separator='\t', quote_char=None, ignore_errors=True)
        
        # 1. Rounds and counts per event
        summary = df.group_by(['competition_id', 'event_id']).agg([
            pl.col('round_type_id').n_unique().alias('r'),
            pl.col('person_id').n_unique().alias('c')
        ])
        
        # 2. Total unique people per competition
        totals = df.group_by('competition_id').agg(
            pl.col('person_id').n_unique().alias('total_c')
        )
        
        cache_data = {"stats": summary.to_dicts(), "totals": totals.to_dicts()}
        with open(RESULTS_CACHE, 'wb') as f:
            f.write(msgpack.packb(cache_data))

def load_cache(path):
    if not os.path.exists(path): return []
    with open(path, 'rb') as f:
        return msgpack.unpackb(f.read())

def get_filtered_data(country, start_date_str, end_date_str):
    try:
        start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
    except: return [], 0, {}

    comps_data = load_cache(COMPS_CACHE)
    if not comps_data: return [], 0, {}
    comps = pl.DataFrame(comps_data)

    filtered_comps = comps.filter(
        (pl.col('country_id') == country) &
        (pl.datetime(pl.col('year'), pl.col('month'), pl.col('day')) >= start_dt) &
        (pl.datetime(pl.col('year'), pl.col('month'), pl.col('day')) <= end_dt)
    ).sort(['year', 'month', 'day'])

    if filtered_comps.is_empty(): return [], 0, {}

    full_res_cache = load_cache(RESULTS_CACHE)
    if not full_res_cache: return [], 0, {}
    
    stats_df = pl.DataFrame(full_res_cache['stats'])
    totals_df = pl.DataFrame(full_res_cache['totals'])

    report, total_rounds, global_event_counts = [], 0, {}
    
    for comp in filtered_comps.to_dicts():
        c_id = comp['id']
        c_stats = stats_df.filter(pl.col('competition_id') == c_id)
        c_total = totals_df.filter(pl.col('competition_id') == c_id)

        if not c_stats.is_empty():
            details = []
            for s in c_stats.to_dicts():
                name = EVENT_NAMES.get(s['event_id'], s['event_id'])
                details.append({'event': name, 'rounds': s['r'], 'count': s['c']})
                total_rounds += s['r']
                global_event_counts[name] = global_event_counts.get(name, 0) + s['r']

            report.append({
                'name': comp['name'],
                'date': f"{comp['year']}-{comp['month']:02d}-{comp['day']:02d}",
                'total_competitors': c_total['total_c'][0] if not c_total.is_empty() else 0,
                'details': sorted(details, key=lambda x: x['event'])
            })

    return report, total_rounds, dict(sorted(global_event_counts.items()))

@app.route('/')
def finder_page():
    pre_cache_data()
    countries = load_cache(COUNTRIES_CACHE)
    country = request.args.get('country', 'Philippines')
    start = request.args.get('start', '2007-01-01')
    end = request.args.get('end', '2026-12-31')
    
    current_iso = next((c['iso2'] for c in countries if c['id'] == country), 'ph')
    data, total_rounds, event_summary = get_filtered_data(country, start, end)
    
    return render_template('finder.html', comps=data, total_rounds=total_rounds, 
                           event_summary=event_summary, country=country, current_iso=current_iso,
                           start=start, end=end, all_countries=countries)

if __name__ == '__main__':
    app.run(debug=True)