import glob, json, os, re, math, collections, statistics as st, bisect
from datetime import date, timedelta

def d2(s):
    y, m, dd = map(int, s.split('-')); return date(y, m, dd)
def clip(x, a=0, b=100): return max(a, min(b, x))

# ---------- ticker -> sector ----------
sec = {}
for f in sorted(glob.glob('data/market/stock_rsi_regime_eval_*.jsonl')):
    for l in open(f):
        if l.strip():
            r = json.loads(l); sec.setdefault(r['ticker'], r.get('sector'))
SECTORS = sorted(set(v for v in sec.values() if v))

# ---------- market snapshots (spine) ----------
snap = {}
for f in glob.glob('data/market/market_snapshot_eval_*.json'):
    dt = re.search(r'(\d{4}-\d\d-\d\d)', os.path.basename(f)).group(1)
    j = json.load(open(f))
    snap[dt] = {r['sector']: r for r in j['sector_rows']}
SPINE = sorted(snap)

Pint = collections.defaultdict(dict); Tnow = collections.defaultdict(dict)
for dt in SPINE:
    for s, r in snap[dt].items():
        Pint[s][dt] = st.mean([r['mk_breadth'], r['rel_perf_breadth'], r['rel_vol_breadth']])
        Tnow[s][dt] = r['T_now']

def near(dt, target_days):
    wd = d2(dt) - timedelta(days=target_days)
    cand = [x for x in SPINE if x <= dt]
    if not cand: return None
    return min(cand, key=lambda x: abs((d2(x) - wd).days))

# ---------- overlay ----------
ovdates = collections.defaultdict(list)
for l in open('data/stock_rsi_regime_overlay_2026.jsonl'):
    if l.strip():
        r = json.loads(l); ovdates[r['date']].append(r)
OVD = sorted(ovdates)
def ov_on_before(dt):
    c = [x for x in OVD if x <= dt]; return c[-1] if c else None

# ---------- prices ----------
def load(path):
    by = collections.defaultdict(list)
    for l in open(path):
        if not l.strip(): continue
        r = json.loads(l)
        if sec.get(r['ticker']) is None: continue
        by[r['ticker']].append(r)
    for t in by: by[t].sort(key=lambda x: x['date'])
    return by
dly = load('data/prices_daily_2026.jsonl')
wk = load('data/prices_weekly_2026.jsonl')
ddates = {t: [b['date'] for b in dly[t]] for t in dly}
TBY = collections.defaultdict(list)
for t in dly:
    if sec.get(t): TBY[sec[t]].append(t)

DIVW = {('w','positive',True):2.0,('w','positive',False):1.0,('w','potential-positive',None):1.0,
('d','positive',True):1.0,('d','positive',False):0.5,('d','potential-positive',None):0.5,
('w','negative',True):-2.0,('w','negative',False):-1.0,('w','potential-negative',None):-1.0,
('d','negative',True):-1.0,('d','negative',False):-0.5,('d','potential-negative',None):-0.5}
def dw(freq, flag, conf):
    if flag in ('positive','negative'): return DIVW.get((freq,flag,conf),0.0)
    if flag in ('potential-positive','potential-negative'): return DIVW.get((freq,flag,None),0.0)
    return 0.0

def asof_idx(t, dt):
    return bisect.bisect_right(ddates[t], dt) - 1

def confirmation(dt):
    out = {}
    for s in SECTORS:
        rs_up = ob_up = spos = sneg = cnt = 0; dwsum = 0.0
        for t in TBY[s]:
            i = asof_idx(t, dt)
            if i < 20: continue
            bars = dly[t]; cnt += 1
            rsv = [bars[k].get('rs') for k in range(i-19,i+1) if bars[k].get('rs') is not None]
            ob = [bars[k].get('obvm') for k in range(i-19,i+1) if bars[k].get('obvm') is not None]
            if rsv and bars[i].get('rs') is not None and bars[i]['rs'] > st.mean(rsv): rs_up += 1
            if ob and bars[i].get('obvm') is not None and bars[i]['obvm'] > st.mean(ob): ob_up += 1
            bt = bars[i].get('last_50_break_type'); bd = bars[i].get('last_50_break_date')
            if bd and bd <= dt:
                bi = bisect.bisect_right(ddates[t], bd) - 1
                if i - bi <= 20:
                    if bt and 'up' in str(bt).lower(): spos += 1
                    elif bt and 'down' in str(bt).lower(): sneg += 1
            dwsum += dw('d', bars[i].get('rsi_divergence_flag'), bars[i].get('rsi_divergence_confirmed'))
        for t in TBY[s]:
            wb = [b for b in wk.get(t, []) if b['date'] <= dt]
            if wb: dwsum += dw('w', wb[-1].get('rsi_divergence_flag'), wb[-1].get('rsi_divergence_confirmed'))
        if cnt < 6: out[s] = None; continue
        rs_b = 100*rs_up/cnt; ob_b = 100*ob_up/cnt
        struct = clip(50 + 1.5*(100*spos/cnt - 100*sneg/cnt))
        dtilt = clip(50 + 10.0*(dwsum/cnt))
        out[s] = 0.35*rs_b + 0.25*ob_b + 0.20*struct + 0.20*dtilt
    return out

def imom_score(dt):
    rows = ovdates[ov_on_before(dt)]
    bys = collections.defaultdict(list)
    for r in rows:
        s = sec.get(r['ticker'])
        if s: bys[s].append(r)
    out = {}
    for s in SECTORS:
        o = bys.get(s, [])
        fl = [x.get('stock_rsi_regime_20d_vs_50d_flag') for x in o]
        scl = [x['stock_rsi_regime_20d_score'] for x in o if isinstance(x.get('stock_rsi_regime_20d_score'),(int,float))]
        if len(fl) < 6 or not scl: out[s] = None; continue
        net = 100*sum(f=='Positive' for f in fl)/len(fl) - 100*sum(f=='Negative' for f in fl)/len(fl)
        cross = clip(50 + 1.5*net)
        breadth = st.mean([100*sum(v>=60 for v in scl)/len(scl), 100-100*sum(v<40 for v in scl)/len(scl), st.mean(scl)])
        out[s] = 0.55*cross + 0.45*breadth
    return out

def label(hlW, tq, infl, imom, conf, fit, GS=20):
    hlS = 100-hlW; rep = 0.35*infl+0.30*imom+0.25*conf+0.10*tq; det = 100-rep
    def g(h, i):
        raw = math.sqrt(max(0,h)*max(0,i)); hg = clip((h-50)/GS,0,1); ig = clip((i-50)/GS,0,1)
        return 50 + (raw-50)*math.sqrt(hg*ig)
    HR = g(hlW, rep); HK = g(hlS, det)
    istrong = infl>=55 and imom>=50 and conf>=55
    iweak = infl<45 or imom<40 or rep<45
    conflicted = (imom>=65 and infl<40) or (imom<=35 and infl>70)
    if HR>=58: lab = ('Contrarian Recovery' if fit<50 else ('Accumulation' if (conf>=58 and imom>=50) else 'Recovery Watch')); tr='HR'
    elif HK>=58: lab='Late Leadership / Distribution Risk'; tr='HK'
    elif hlS>=60 and imom<=35: lab='Distribution Watch (early)'; tr='rollover'
    elif hlW>=60 and imom>=65 and infl>=50: lab='Recovery Watch'; tr='repair'
    elif conflicted: lab='Neutral / No Edge'; tr='conflict'
    elif hlS>=70 and istrong: lab='Continuation Candidate'; tr='agree'
    elif hlS>=60 and not iweak: lab='Healthy Leadership'; tr='agree'
    elif hlW>=60 and iweak: lab='Deteriorating'; tr='agree'
    else: lab='Neutral / No Edge'; tr='no edge'
    return lab, tr, HR, HK, rep

def fwd_return(dt):
    tgt = d2(dt).toordinal() + 28
    out = {}
    for s in SECTORS:
        rets = []
        for t in TBY[s]:
            i = asof_idx(t, dt)
            if i < 0: continue
            arr = ddates[t]; j = None
            for k in range(i+1, len(arr)):
                if d2(arr[k]).toordinal() >= tgt: j = k; break
            if j is None: continue
            c0 = dly[t][i].get('adjusted_close'); c1 = dly[t][j].get('adjusted_close')
            if c0 and c1: rets.append(c1/c0 - 1)
        out[s] = st.mean(rets) if len(rets) >= 6 else None
    return out

# ---------- run ----------
records = []
for dt in SPINE:
    conf = confirmation(dt); im = imom_score(dt); fwd = fwd_return(dt)
    onem = {s: snap[dt][s]['one_month_pct'] for s in SECTORS if s in snap[dt]}
    order = sorted(onem, key=lambda s: onem[s]); n = len(order)
    hlS = {s: 100*order.index(s)/(n-1) for s in order}
    d1w = near(dt, 7); d1m = near(dt, 30)
    rows_dt = {}
    for s in SECTORS:
        if s not in snap[dt] or conf.get(s) is None or im.get(s) is None: continue
        r = snap[dt][s]; Pi = Pint[s][dt]
        Pi1w = Pint[s].get(d1w, Pi); Pi1m = Pint[s].get(d1m, Pi)
        T1w = Tnow[s].get(d1w, r['T_now']); T1m = Tnow[s].get(d1m, r['T_now'])
        dPi = 0.6*(Pi-Pi1w)+0.4*((Pi-Pi1m)/4); dT = 0.6*(r['T_now']-T1w)+0.4*((r['T_now']-T1m)/4)
        infl = clip(50 + 8*st.mean([dPi, dT]))
        hlw = 100 - hlS[s]
        rec = dict(dt=dt, s=s, infl=infl, imom=im[s], conf=conf[s], tq=r['T_now'], hlw=hlw,
                   fit=r.get('sector_regime_fit_score', 60), fwd=fwd.get(s), onem=onem[s])
        records.append(rec); rows_dt[s] = rec
    fl = [(s, fwd.get(s)) for s in rows_dt if fwd.get(s) is not None]
    fl.sort(key=lambda x: x[1])
    fr = {s: i/(len(fl)-1) for i,(s,_) in enumerate(fl)} if len(fl) > 1 else {}
    for s, rec in rows_dt.items():
        rec['fwdrank'] = fr.get(s)

json.dump(records, open('data/_harness_records.json','w'))
print('records:', len(records), 'across', len(SPINE), 'dates', SPINE[0], '->', SPINE[-1])
print('with forward outcome:', sum(1 for r in records if r['fwd'] is not None))

def relabel(records, GS):
    for r in records:
        lab, tr, HR, HK, rep = label(r['hlw'], r['tq'], r['infl'], r['imom'], r['conf'], r['fit'], GS)
        r['lab'], r['tr'], r['HR'], r['HK'] = lab, tr, HR, HK

def pctl(a, p):
    a = sorted(a); return a[int(p*(len(a)-1))]

def success(r):
    f = r['fwd']; rk = r['fwdrank']
    if f is None or rk is None: return None
    lab = r['lab']
    if lab in ('Continuation Candidate','Healthy Leadership','Recovery Watch','Accumulation','Contrarian Recovery'):
        return rk >= 0.5 or f > 0
    if lab in ('Late Leadership / Distribution Risk','Distribution Watch (early)'):
        return rk < 0.5 or f < 0
    if lab == 'Deteriorating':
        return rk < 0.5
    return None

from collections import Counter, defaultdict
for GS in (20, 15, 12):
    relabel(records, GS)
    HRs = [r['HR'] for r in records]; HKs = [r['HK'] for r in records]
    nfire = sum(r['HR']>=58 or r['HK']>=58 for r in records)
    print('\n========== gate_span = %d =========='%GS)
    print('HR/HK != 50: %.0f%% / %.0f%%   HR p90 %.0f max %.0f   HK p90 %.0f max %.0f   #(HR>=58 or HK>=58)=%d'%(
        100*sum(abs(x-50)>0.5 for x in HRs)/len(HRs), 100*sum(abs(x-50)>0.5 for x in HKs)/len(HKs),
        pctl(HRs,.9), max(HRs), pctl(HKs,.9), max(HKs), nfire))
    cnt = Counter(r['lab'] for r in records)
    hit = defaultdict(lambda: [0,0])
    for r in records:
        sc = success(r)
        if sc is None: continue
        hit[r['lab']][0] += sc; hit[r['lab']][1] += 1
    print('%-36s count scored hit%%'%'label')
    for lab, c in cnt.most_common():
        h, nn = hit[lab]
        print('%-36s %5d %6d  %s'%(lab, c, nn, ('%.0f%%'%(100*h/nn)) if nn else '-'))
