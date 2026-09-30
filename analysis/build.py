#!/usr/bin/env python3
"""Rebuild the standalone field report from local transcript data. Stdlib only."""
import sys, json, re, math, hashlib, random, statistics
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit, urlencode
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import viewer
ROOT = viewer.ROOT
OUT = ROOT / 'analysis'
if '--render-only' in sys.argv:
 payload=(OUT/'results.json').read_text().replace('<','\\u003c')
 (ROOT/'analysis.html').write_text((OUT/'report-template.html').read_text().replace('__ANALYSIS_DATA__',payload))
 print('Rendered analysis.html from saved results')
 sys.exit(0)
RULES = {
 'VERIFY': r'\b(verif(?:y|ied|ication)|independently check(?:ed)?|cross[- ]check(?:ed)?|confirmed live|byte[- ]exact)\b',
 'PROPOSE': r'\b(i propose|i suggest|we should|let[’\x27]s|how about|recommend(?:ation)?)\b',
 'COORDINATE': r'\b(please (?:check|review|verify|run|publish|take|update)|can you|could you|ready for (?:review|verification)|you[’\x27]re unblocked|assign(?:ed)?|volunteer|join me)\b',
 'ACKNOWLEDGE': r'\b(agreed|received|acknowledged|thanks|thank you|i[’\x27]ll (?:follow|use|adopt)|following your)\b',
 'CHALLENGE': r'\b(disagree|incorrect|not correct|unsupported|unverified|contradict(?:s|ion)?|reliability issue|breach|please exclude)\b',
 'BLOCKED': r'\b(blocked|stuck|unable to|cannot access|can[’\x27]t access|permission denied|timed out|404|propagat(?:ing|ion))\b',
 'REPORT': r'\b(published|deployed|shipped|merged|completed|committed|is live|pushed|finished)\b',
 'RESEARCH': r'\b(research(?:ing|ed)?|investigat(?:e|ing|ed)|look(?:ing)? into|search(?:ing|ed)? for)\b',
 'ABANDON': r'\b(abandon(?:ed|ing)?|dropping (?:this|the)|stop pursuing|shelv(?:e|ed|ing)|no longer pursue)\b'
}
compiled = {k:re.compile(v, re.I) for k,v in RULES.items()}
URL = re.compile(r'https?://[^\s<>\[\]"`]+')
STOP = set('the a an and or for to of in is it this that with on as at by from be are was were i we you my our your has have had not no will can all now new its their they them but if so into more same also than then been about just out up one two three each any do does did me us he she his her there here these those some only still today update thanks thank please day next ready goal session agent village http https com org www'.split())
def actor(e): return e.get('speakerName') or e.get('agentName') or 'Unknown'
def text(e): return str(e.get('content') or e.get('message') or '')
def stamp(e):
 try: return datetime.fromisoformat(e['timestamp'].replace('Z','+00:00')).timestamp()
 except (KeyError, ValueError): return None

def evidence(day, i, e):
 t=text(e)
 return dict(day=day['day'], date=day['date'], index=i, speaker=actor(e), time=e.get('time',''), timestamp=e.get('timestamp',''), text=t[:2800], truncated=len(t)>2800, link='http://127.0.0.1:8765/?'+urlencode({'day':day['day'],'type':e.get('type',''),'q':t[:110]}))
def entropy(c):
 total=sum(c.values())
 return -sum((n/total)*math.log(n/total) for n in c.values() if n) if total else 0

def jsd(a,b):
 ta,tb=sum(a.values()),sum(b.values())
 if not ta or not tb:return 0
 result=0
 for k in a.keys()|b.keys():
  p,q=a.get(k,0)/ta,b.get(k,0)/tb;m=(p+q)/2
  if p:result+=p*math.log2(p/m)/2
  if q:result+=q*math.log2(q/m)/2
 return result

print('Discovering agent names…',flush=True)
registry=set()
for day,_,_ in viewer.load_days():
 for e in day.get('events',[]):
  if e.get('type')=='AGENT_TALK' or e.get('agentName'): registry.add(actor(e))
registry.discard('Unknown')
# Longest name first and explicit boundaries avoid GPT-5 matching GPT-5.1.
name_re=re.compile(r'(?<![\w-])('+ '|'.join(re.escape(n) for n in sorted(registry,key=len,reverse=True))+r')(?![\w-]|\.\w)',re.I)
canonical={n.casefold():n for n in registry}
rows=[];types=Counter();labels=Counter();global_edges=Counter();agent_counts=Counter();agent_labels=defaultdict(Counter)
artifacts=[];duplicates=[];all_examples=defaultdict(list);total=0;human=0;messages=0;covered=0;repeated=0;edge_examples={};allprofiles=[];month=defaultdict(Counter)
audit_rng=random.Random(732);audit_pools=defaultdict(list);audit_seen=Counter()
def sample_review(tag, ev, limit):
 audit_seen[tag]+=1
 if len(audit_pools[tag])<limit:audit_pools[tag].append(ev)
 else:
  pos=audit_rng.randrange(audit_seen[tag])
  if pos<limit:audit_pools[tag][pos]=ev
print('Measuring messages, references and artifact recurrences…',flush=True)
for day,_,_ in viewer.load_days():
 events=day.get('events',[]);total+=len(events);dc=Counter(e.get('type','UNKNOWN') for e in events);types.update(dc)
 talks=[(i,e) for i,e in enumerate(events) if e.get('type')=='AGENT_TALK'];human+=dc['USER_TALK'];messages+=len(talks)
 speakers=Counter(actor(e) for _,e in talks);active=set(speakers);edges=Counter();lc=Counter();examples=defaultdict(list)
 lexical=defaultdict(Counter);same=defaultdict(list);urls=defaultdict(list);times=defaultdict(list);daycovered=0
 for i,e in talks:
  who=actor(e);t=text(e);ts=stamp(e);ev=evidence(day,i,e)
  agent_counts[who]+=1
  tags=[k for k,pat in compiled.items() if pat.search(t)]
  lc.update(tags);labels.update(tags);agent_labels[who].update(tags)
  if tags:covered+=1;daycovered+=1
  else:sample_review('NO_CUE',ev,24)
  for tag in tags:
   sample_review(tag,ev,12)
   if len(examples[tag])<2:examples[tag].append(ev)
   if len(all_examples[tag])<4:all_examples[tag].append(ev)
  for match in {canonical[m.group(1).casefold()] for m in name_re.finditer(t)}-{who}:
   global_edges[who,match]+=1
   if match in active:edges[who,match]+=1
   edge_examples.setdefault((who,match),ev)
  tokens=[w for w in re.findall(r'[a-z]{3,}', URL.sub('',t).lower()) if w not in STOP]
  lexical[who].update(tokens)
  normalized=' '.join(t.casefold().split())
  if len(normalized)>=80:same[normalized].append(ev)
  if ts is not None:
   times[who].append(ts)
   for u in set(URL.findall(t)):
    u=u.rstrip('.,;:!?)]}’\x27')
    try:
     p=urlsplit(u)
     if len(p.path)<6:continue
     u=urlunsplit((p.scheme.lower(),p.netloc.lower(),p.path,p.query,''))
     urls[u].append((ts,who,ev))
    except ValueError:pass
 for norm,items in same.items():
  if len(items)>1:
   repeated+=len(items)-1
   duplicates.append(dict(day=day['day'],date=day['date'],count=len(items),agents=len(set(e['speaker'] for e in items)),characters=len(norm),examples=items[:6]))
 for u,items in urls.items():
  first={}
  for ts,who,ev in sorted(items,key=lambda v:v[0]):first.setdefault(who,(ts,ev))
  if len(first)<3:continue
  seq=sorted(first.items(),key=lambda v:v[1][0]);span=(seq[-1][1][0]-seq[0][1][0])/60
  if span>60:continue
  artifacts.append(dict(day=day['day'],date=day['date'],url=u,agents=len(first),mentions=len(items),minutes=round(span,2),active=len(active),sequence=[dict(minutes=round((v[0]-seq[0][1][0])/60,2),**v[1]) for who,v in seq]))
 neighbors={n:set() for n in active}
 for a,b in edges:neighbors[a].add(b);neighbors[b].add(a)
 unseen=set(active);sizes=[]
 while unseen:
  stack=[unseen.pop()];size=0
  while stack:
   n=stack.pop();size+=1
   for v in neighbors[n]&unseen:unseen.remove(v);stack.append(v)
  sizes.append(size)
 concentration=sum((n/len(talks))**2 for n in speakers.values()) if talks else None
 row=dict(day=day['day'],date=day['date'],events=len(events),messages=len(talks),active=len(active),human=dc['USER_TALK'],types=dict(dc),labels=dict(lc),covered=daycovered,speakers=dict(speakers),edges=[dict(source=a,target=b,count=n) for (a,b),n in edges.items()],largestComponent=max(sizes,default=0),components=len(sizes),effectiveSpeakers=round(1/concentration,2) if concentration else None,examples=dict(examples))
 profiles=[]
 for who,c in lexical.items():
  if sum(c.values())>=20:
   profile=dict(c.most_common(120));profiles.append(profile);allprofiles.append(profile)
 row['_profiles']=profiles
 rows.append(row);month[day['date'][:7]].update(dc)
 if len(rows)%100==0:print(f'  {len(rows)} days processed',flush=True)
# TF-IDF cosine on per-agent daily lexical profiles; no semantic embeddings.
df=Counter(w for p in allprofiles for w in p);idf={w:math.log((1+len(allprofiles))/(1+n))+1 for w,n in df.items()}
for r in rows:
 vectors=[]
 for p in r.pop('_profiles'):
  v={w:(1+math.log(n))*idf[w] for w,n in p.items()};norm=math.sqrt(sum(n*n for n in v.values()));vectors.append({w:n/norm for w,n in v.items()})
 vals=[]
 for i,a in enumerate(vectors):
  for b in vectors[i+1:]:vals.append(sum(v*b.get(w,0) for w,v in a.items()))
 r['topicOverlap']=round(statistics.mean(vals),4) if vals else None
 r['topicPairs']=len(vals)
# Exported changelog context, preserving wording as dataset documentation.
changes=[]
for section in re.split(r'^## ', (ROOT/'CHANGELOG.md').read_text(),flags=re.M)[1:]:
 heading,*body=section.split('\n',1)
 if re.match(r'\d{4}-\d{2}-\d{2}',heading):changes.append(dict(date=heading[:10],title=heading,notes=body[0].strip() if body else ''))
shifts=[];previous=None
for r in rows:
 if r['events']>=100:
  if previous:
   delta=jsd(previous['types'],r['types'])
   nearby=[c for c in changes if abs((datetime.fromisoformat(c['date'])-datetime.fromisoformat(r['date'])).days)<=7]
   shifts.append(dict(day=r['day'],date=r['date'],previousDay=previous['day'],previousDate=previous['date'],score=round(delta,4),before=previous['types'],after=r['types'],context=nearby[:6]))
  previous=r
artifacts.sort(key=lambda x:(-x['agents'],x['date'],x['url']))
duplicates.sort(key=lambda x:(-x['count'],-x['agents'],x['date']))
shifts.sort(key=lambda x:-x['score'])
# Regime comparison, normalizing within each event stream (not productivity).
periods=[]
for title,lo,hi in [('Session era', '0000','2026-03-11'),('Rollout window','2026-03-11','2026-03-24'),('Permanent computer use','2026-03-24','9999')]:
 rs=[r for r in rows if lo<=r['date']<hi];cnt=Counter()
 for r in rs:cnt.update(r['types'])
 periods.append(dict(title=title,days=len(rs),events=sum(cnt.values()),types=dict(cnt),messages=sum(r['messages'] for r in rs),meanActive=round(statistics.mean(r['active'] for r in rs if r['messages']),2)))
# Review sample for future human coding. No model labels or calibrated probabilities.
audit=[dict(proposedLabel=tag,reviewLabel='',**e) for tag,pool in audit_pools.items() for e in pool]
digest=hashlib.sha256()
with viewer.TRANSCRIPT.open('rb') as source:
 for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk)
result=dict(title='AI Village · Computational ethology',methodVersion='1.0',source=dict(file='village-transcript.json',bytes=viewer.TRANSCRIPT.stat().st_size,sha256=digest.hexdigest(),exportedAt=json.loads((ROOT/'manifest.json').read_text())['exportedAt']),summary=dict(days=len(rows),nonemptyDays=sum(bool(r['events']) for r in rows),first=rows[0]['date'],last=rows[-1]['date'],events=total,messages=messages,human=human,agents=len(agent_counts),types=dict(types),labels=dict(labels),covered=covered,repeatedMessages=repeated,duplicateGroups=len(duplicates),artifactCandidates=len(artifacts)),days=rows,agents=[dict(name=n,messages=c,labels=dict(agent_labels[n])) for n,c in agent_counts.most_common()],edges=[dict(source=a,target=b,count=n,example=edge_examples[a,b]) for (a,b),n in global_edges.most_common(100)],artifacts=artifacts[:24],duplicates=duplicates[:16],shifts=shifts[:12],periods=periods,changes=changes,rules=RULES)
if (OUT/'findings.json').exists():
 findings=json.loads((OUT/'findings.json').read_text())
 # Hydrate abbreviated research notes with verbatim source excerpts.
 entries={d['day']:d for d in viewer.day_index()['days']}
 for finding in findings['findings']:
  for i,ev in enumerate(finding.get('sourceEvidence',[])):
   day=viewer.read_day(entries[ev['day']])
   finding['sourceEvidence'][i]=evidence(day,ev['index'],day['events'][ev['index']])
 result['findings']=findings
result['reviewSample']=audit
result['summary']['uniqueDates']=len(set(r['date'] for r in rows))
OUT.mkdir(exist_ok=True)
(OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')))
(OUT/'review-sample.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
if (OUT/'report-template.html').exists():
 template=(OUT/'report-template.html').read_text()
 payload=json.dumps(result,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
 (ROOT/'analysis.html').write_text(template.replace('__ANALYSIS_DATA__',payload))
print(json.dumps(result['summary'],indent=2))
print('Top artifacts:',[(a['day'],a['agents'],a['minutes'],a['url']) for a in artifacts[:4]])
print('Top shifts:',[(s['day'],s['date'],s['score']) for s in shifts[:5]])
print('Wrote analysis.html, analysis/results.json and analysis/review-sample.json',flush=True)
