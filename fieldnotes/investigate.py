#!/usr/bin/env python3
"""Rebuild the three locally curated FIELDNOTES episodes from immutable day records.
Run from the ai-village directory: python3 fieldnotes/investigate.py"""
import hashlib, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import viewer

OUT = Path(__file__).with_name('investigations.json')
# Frozen editorial selections; each locator is (day, zero-based event index).
CURATION = [
 {"id":"day315-form-lineage","day":315,"title":"A working-link report becomes an IP-blocking explanation","question":"Which parts of the Google Form story were supported by a human access report, agent checks, a deployment report, or a disputed repository inspection?","phenomenon":"idea-lineage","locators":[(315,i) for i in [357,358,564,567,569,570,571,572,580,583,585,587,599,604,615,617]],"claims":[
  {"id":"c1","text":"A human contributor was reported as saying the short URL worked on multiple computers with friends; DeepSeek-V3.2 separately reported HTTP 200 and form elements.","qualifier":"The human message is quoted secondhand in this transcript. DeepSeek's check is an agent report, not reproduced here."},
  {"id":"c2","text":"DeepSeek-V3.2 relayed a secondhand explanation that Google's auth/IP filtering might block agents, describing it as a suggestion. Gemini 3 Pro framed agent-IP blocking conditionally; Opus 4.5 later stated it as fact.","qualifier":"The cited source is a human explanation relayed by an agent. No controlled comparison or independent network evidence demonstrates the cause."},
  {"id":"c3","text":"Claude Opus 4.6 reported copying the URL into the site in commit 22c07da. Later agents reported seeing it in main-branch HTML and on GitHub Pages.","qualifier":"The reported checks establish code/CTA presence and link destination, not successful form submission."},
  {"id":"c4","text":"Reports conflict over whether TEST_URL buttons existed on the main branch before the integration: DeepSeek reported finding them; Sonnet later reported they were absent.","qualifier":"The branch, time, and inspected source differed or may have differed; keep the discrepancy unresolved."},
  {"id":"c5","text":"Gemini 2.5 Pro reported that an earlier form build was blocked by a GUI bug and had to be restarted.","qualifier":"This is an agent's account of its form-building attempt and does not identify the later URL's cause."}],
  "rootLabel":"Human-access report, as quoted by DeepSeek-V3.2","rootEvents":["d315-e564"],"extraRoots":[{"id":"day315-form-lineage-root2","label":"Reported site integration commit 22c07da","basis":"Opus 4.6 reports this commit as its own integration action; later speakers explicitly attribute live-button statements to that commit.","eventIds":["d315-e583"],"caveat":"This groups attributed reports about the integration only. Subsequent code/live checks remain separate, and no form submission outcome is documented."}],
 "edgeSpecs":[("d315-e564","d315-e569","attributes_to","Haiku's plan explicitly credits DeepSeek-V3.2 and bearsharktopus-dev for the verified-link report.","explicit-attribution",["d315-e564","d315-e569"],["The human's original Issue #8 comment is absent; both records report it secondhand."]),("d315-e564","d315-e570","mentions","Gemini 3 Pro directly addresses DeepSeek and refers to bearsharktopus-dev's finding, but phrases agent-IP blocking conditionally.","explicit-attribution",["d315-e564","d315-e570"],["The human comment itself is absent; only DeepSeek's quotation of it is logged.","The proposed IP explanation is conditional, not verified."]),("d315-e564","d315-e571","attributes_to","Opus 4.5 explicitly identifies bearsharktopus-dev as the source of the verified-link report.","explicit-attribution",["d315-e564","d315-e571"],["The report that no CTA was visible concerns site state before the later reported integration."]),("d315-e564","d315-e572","attributes_to","Opus 4.5 (Claude Code) explicitly calls the URL human-verified and names DeepSeek/Haiku in its follow-up; it turns the access-context explanation into an assertion.","explicit-attribution",["d315-e564","d315-e572"],["The transcript does not contain the human's original comment or a controlled comparison of access contexts."]),("d315-e564","d315-e580","attributes_to","Claude 3.7 Sonnet says it confirmed DeepSeek's discovery and repeats that the link was verified by bearsharktopus-dev and multiple humans.","explicit-attribution",["d315-e564","d315-e580"],["The report does not contain the original human message or independent human test records."]),("d315-e583","d315-e585","attributes_to","Opus 4.5 (Claude Code) explicitly names Claude Opus 4.6's commit 22c07da as the integration source.","explicit-attribution",["d315-e583","d315-e585"],["Both events report repository state; no repository state was independently fetched here."]),("d315-e583","d315-e587","attributes_to","Haiku explicitly attributes its live-integration statement to Claude Opus 4.6 and the reported commit.","explicit-attribution",["d315-e583","d315-e587"],["Haiku says it learned this from chat history; this is attribution, not its own live-page check."]),("d315-e567","d315-e604","challenges","Sonnet 4.5's earlier event reports the TEST_URL discrepancy; its later event explicitly calls DeepSeek's three-button report incorrect and describes searching main.","explicit-attribution",["d315-e567","d315-e604"],["The inspected branch or moment may differ from the one in DeepSeek's report."])],
  "artifacts":[("Google Form","https://forms.gle/6ZNTydyA2rwZyq6V7","reported-working",["human access reported","agent HTTP 200 reported","site integration reported","live-page fetch reported"],["d315-e564","d315-e570","d315-e583","d315-e615"],"No external content fetched for this investigation; live-site evidence is a quoted transcript report.")],
  "alternatives":["The link may have worked for the human contributor while agent IPs encountered a different response; no controlled comparison appears here.","The old malformed link may have resulted from URL copy/paste mangling, as Opus 4.6 later reports, rather than agent-IP filtering.","TEST_URL findings may refer to a stale branch or differing inspection time; Sonnet reports that main lacked those strings."],
  "counter":[("d315-e570","Gemini 3 Pro frames agent-IP blocking as conditional ('if it blocks some agent IPs'); this weakens Opus 4.5's later categorical wording."),("d315-e604","Claude Sonnet 4.5 reports that main never contained the three alleged TEST_URL buttons, challenging DeepSeek's earlier diagnosis."),("d315-e617","Claude Sonnet 4.5 again reports no TEST_URL buttons on main and says the URL was added later, preserving the source-state conflict.")],
  "summary":"The transcript distinguishes a secondhand human-access report, DeepSeek's own reported HTTP check, a tentative and then assertive IP-filtering explanation, a reported site edit, and later reports of CTA presence. Opus 4.6 also names URL copy/paste mangling as an explanation for the earlier broken link. The site checks establish reported link presence, not a successful form submission. A separate TEST_URL disagreement remains visible.","scope":{"start":357,"end":617,"prehistory":"Whole-corpus exact URL scan before the first selected claim; see method.prehistoryScan.","coverage":"Selected Day 315 transcript events only; external form/site not fetched."}},
 {"id":"day321-capsule-boundary","day":321,"title":"A time-capsule repository with disputed pull-request state","question":"What can the transcript establish about the repository and its early contributions when reported PR state conflicts?","phenomenon":"boundary-case","locators":[(321,i) for i in [58,59,64,67,69,72,76,95,97,101,107,120,128,132,135,153]],"claims":[
  {"id":"c1","text":"Claude Sonnet 4.5 reported creating the village-time-capsule repository; other agents described intended or reported contributions.","qualifier":"Repository existence and action are agent-reported here."},
  {"id":"c2","text":"Opus 4.5 (Claude Code) reported PR #1 created, while Claude Sonnet 4.5 later reported PR #1 did not exist when it attempted review.","qualifier":"Conflicting live-state reports; no repository state is independently inspected here."},
  {"id":"c3","text":"A later message says two PRs were live, but that later claim does not resolve the earlier discrepancy by itself.","qualifier":"Chronological follow-up is not proof of which account was accurate at the earlier moment."}],
  "rootLabel":"Reported repository creation","rootEvents":["d321-e58"],"edgeSpecs":[("d321-e58","d321-e67","references_artifact","PR announcement says PR #1 was created for the named repository and asks its creator to review.","explicit-attribution",["d321-e58","d321-e67"],["A transcript report does not independently establish remote PR state."]), ("d321-e67","d321-e95","challenges","Both events concern PR #1; Sonnet 4.5 explicitly says it does not exist when checked, conflicting with the earlier creation announcement.","explicit-attribution",["d321-e67","d321-e95"],["Remote state may have changed between checks.","One or both reports may be mistaken or refer to different account/repository contexts."]), ("d321-e132","d321-e153","challenges","Opus 4.5 reports all three PRs exist; Sonnet 4.5 later says PRs #1–5 do not exist despite announcements.","explicit-attribution",["d321-e132","d321-e153"],["The events may reflect different check times or GitHub visibility/account context."])],
  "artifacts":[("Village Time Capsule repository","https://github.com/ai-village-agents/village-time-capsule","reported-created",["repository creation reported","PR #1 creation reported","PR #1 absent reported","PR #2 creation reported"],["d321-e58","d321-e67","d321-e95","d321-e101"],"No external fetch; repository and PR lifecycle are described in transcript reports.")],
  "alternatives":["PR #1 may have become visible after an indexing or account-context delay.","The pull request may have been created after Sonnet 4.5's check, despite source ordering; source order is not reliable wall-clock proof for these reports.","The reported repository may have had visibility or permission differences across agents."],
  "counter":[("d321-e95","Claude Sonnet 4.5 reports zero open and zero closed PRs and says PR #1 does not exist."),("d321-e107","Claude Sonnet 4.5 later says both PRs are live; this demonstrates continued reported status change but does not independently resolve the earlier contradiction."),("d321-e153","Sonnet 4.5 reports a troubleshooting attempt against PR #5 returned a GraphQL not-found error, a separate boundary on accessible remote state.")],
  "summary":"The time-capsule episode is useful as a boundary case: a repository and contributions are repeatedly reported, but one named PR is reported both present and absent. The data supports analyzing visibility and check timing, not declaring either account correct.","scope":{"start":58,"end":153,"prehistory":"Whole-corpus exact repository URL scan before Day 321 event 58.","coverage":"Selected Day 321 chat reports; no GitHub API or repository fetch."}},
 {"id":"day454-pattern-archive","day":454,"title":"Pattern submissions move through review artifacts","question":"How did a pattern contribution move through files, merge request review, and reported validation?","phenomenon":"artifact-pathway","locators":[(454,i) for i in [665,669,672,673,674,728,739,740,743,1145,1161,1162,1205,1206,1274,1280,1281,1289,1294]],"claims":[
  {"id":"c1","text":"GPT-5.2 reported that MR !6 added a pattern, metadata, and a README catalog entry; DeepSeek-V3.2 acknowledged this as a repository-based submission.","qualifier":"Submission and review status are transcript reports."},
  {"id":"c2","text":"A separate Pages-drift contribution was reported in MR !18, later reported merged, with subsequent consistency and deployment checks reported.","qualifier":"Merge and validation are agent-reported; no remote artifact was fetched here."},
  {"id":"c3","text":"MR !19 was reported ready, then merged, then checked; a later note raised JSON companion consistency as a separate follow-up.","qualifier":"Do not conflate the README checker passing with every Markdown file having a JSON companion."}],
  "rootLabel":"MR !6 pattern submission as reported","rootEvents":["d454-e665"],"edgeSpecs":[("d454-e665","d454-e669","reports_result","GPT-5.2 reports a JSON metadata fix on the same MR branch.","explicit-attribution",["d454-e665","d454-e669"],["Change is reported; content hash is not locally available."]), ("d454-e669","d454-e672","reports_result","GPT-5.2 reports updating the README on the same MR branch.","explicit-attribution",["d454-e669","d454-e672"],["Change is reported; branch artifact not fetched."]), ("d454-e1145","d454-e1162","reports_result","GPT-5.2 reports MR !18, then later says it merged successfully.","explicit-attribution",["d454-e1145","d454-e1162"],["Merge state is not independently verified in this investigation."]), ("d454-e1274","d454-e1280","reports_result","GPT-5.2 reports MR !19 ready, then reports merging it.","explicit-attribution",["d454-e1274","d454-e1280"],["The reported merge does not prove the wider pattern metric claim."]), ("d454-e1280","d454-e1281","reports_result","The same speaker reports post-merge consistency validation at the merge commit.","explicit-attribution",["d454-e1281"],["A consistency checker result is not a content quality review."])],
  "artifacts":[("MR !6 pattern contribution","https://gitlab.com/ai-village-agents/village/deepseek-pattern-archive/-/merge_requests/6","reported-merged-or-submitted",["MR created reported","metadata amended reported","README updated reported","review/mergeable status reported"],["d454-e665","d454-e669","d454-e672","d454-e674"],"This sample does not include an explicit final merge report for MR !6."),("MR !18 Pages-drift pattern","https://gitlab.com/ai-village-agents/village/deepseek-pattern-archive/-/merge_requests/18","reported-merged",["MR opened reported","pipeline/check reported","merge reported","post-merge validation reported"],["d454-e1145","d454-e1161","d454-e1162","d454-e1206"],"Remote state and check output are reported, not fetched."),("MR !19 recovery pattern","https://gitlab.com/ai-village-agents/village/deepseek-pattern-archive/-/merge_requests/19","reported-merged",["ready reported","merge reported","post-merge checker reported","JSON companions follow-up opened"],["d454-e1274","d454-e1280","d454-e1281","d454-e1294"],"Later message says follow-up MR !20 adds JSON companions; its merge outcome is outside this sample.")],
  "alternatives":["A green pipeline and README checker can validate structural rules without assessing whether the pattern is accurate or useful.","Metrics such as adoption/application are self-reported and definitions may shift; they are not direct evidence of downstream efficacy.","The archive's master branch and GitHub mirror reportedly diverged, so artifact identity and deployment target matter."],
  "counter":[("d454-e740","GPT-5.2 reports MR !5 was mergeable but did not add a patterns entry or update the README count, challenging any assumption that every MR represented a pattern submission."),("d454-e743","GPT-5.2 reports the pattern-file count and README count disagreed."),("d454-e1289","GPT-5.2 reports README checker success while noting a naïve scan found Markdown-only files; checker success does not prove companion completeness.")],
  "summary":"Day 454 shows an artifact workflow in which contributions are described as Markdown/JSON files plus catalog updates, reviewed through merge requests, and checked with scripts. The transcript also exposes limits: some MRs were analysis-only, counts drifted, and a checker could pass despite missing companions.","scope":{"start":665,"end":1294,"prehistory":"Whole-corpus exact archive URL/repository-name scan before first selected MR report.","coverage":"Selected Day 454 reports only; no GitLab/GitHub fetch."}}
]

# Per-event judgments are frozen separately from extraction. Each tuple is
# (track, claim, stance, outcome, documented_check, endorsement, root_family).
def A(track, claim, stance, outcome, check=False, endorsement=False, root=None):
    return {"track":track,"claimId":claim,"stance":stance,"outcomeStatus":outcome,
            "isDocumentedCheck":check,"isEndorsement":endorsement,"rootFamily":root}

ANN = {
315:{
357:A('action','c5','report','proposed'),
358:A('expression','c5','report','result-reported'),
564:A('evidence','c1','endorsement','result-reported',True,True,'day315-form-lineage-root1'),
567:A('expression','c4','challenge','unknown',False),
569:A('expression','c1','endorsement','result-reported',False,True,'day315-form-lineage-root1'),
570:A('expression','c2','qualification','unknown',True,False,'day315-form-lineage-root1'),
571:A('evidence','c1','report','result-reported',True,False,'day315-form-lineage-root1'),
572:A('expression','c2','endorsement','result-reported',False,True,'day315-form-lineage-root1'),
580:A('evidence','c1','endorsement','result-reported',True,True,'day315-form-lineage-root1'),
583:A('action','c3','endorsement','result-reported',True,True,'day315-form-lineage-root2'),
585:A('action','c3','endorsement','result-reported',False,True,'day315-form-lineage-root2'),
587:A('expression','c3','endorsement','result-reported',False,True,'day315-form-lineage-root2'),
599:A('evidence','c3','endorsement','result-reported',True,True),
604:A('evidence','c4','challenge','result-reported',True),
615:A('evidence','c3','endorsement','result-reported',True,True),
617:A('evidence','c4','challenge','result-reported',False),
},
321:{
58:A('action','c1','report','result-reported'),
59:A('expression','c1','endorsement','proposed',False,True),
64:A('context','c1','report','result-reported',True),
67:A('action','c2','report','result-reported'),
69:A('expression','c1','endorsement','proposed',False,True),
72:A('expression','c1','endorsement','proposed',False,True),
76:A('action','c2','report','proposed'),
95:A('evidence','c2','challenge','contradicted',True),
97:A('context','c1','report','proposed'),
101:A('action','c2','report','result-reported'),
107:A('expression','c2','endorsement','result-reported',False,True),
120:A('expression','c1','report','proposed'),
128:A('action','c1','report','result-reported'),
132:A('evidence','c2','endorsement','result-reported',True,True),
135:A('evidence','c2','qualification','unknown',True),
153:A('evidence','c2','challenge','contradicted',True),
},
454:{
665:A('action','c1','report','result-reported'),
669:A('action','c1','report','result-reported'),
672:A('action','c1','report','result-reported'),
673:A('expression','c1','endorsement','result-reported',False,True),
674:A('evidence','c1','qualification','result-reported',True),
728:A('evidence','c1','report','result-reported',True),
739:A('evidence','c1','report','result-reported',True),
740:A('evidence','c1','qualification','result-reported',True),
743:A('evidence','c1','challenge','result-reported',True),
1145:A('action','c2','report','result-reported'),
1161:A('evidence','c2','report','result-reported',True),
1162:A('action','c2','report','result-reported'),
1205:A('expression','c2','endorsement','result-reported',False,True),
1206:A('evidence','c2','qualification','result-reported',True),
1274:A('evidence','c3','report','result-reported',True),
1280:A('action','c3','report','result-reported'),
1281:A('evidence','c3','report','result-reported',True),
1289:A('evidence','c3','qualification','result-reported',True),
1294:A('action','c3','report','result-reported'),
}}

QUOTE_MARKERS = {
 (315,564):'works on multiple computers with friends',
 (315,569):'verified by DeepSeek-V3.2, bearsharktopus-dev, and multiple humans',
 (315,570):'even if it blocks some agent IPs',
 (315,572):'Google was blocking our agent IPs, not the actual form',
 (315,571):'@bearsharktopus-dev found a **VERIFIED WORKING** Google Form',
 (315,580):"I confirmed DeepSeek-V3.2's discovery that the form URL",
 (315,583):'**Commit 2 (`22c07da`): Google Form integration**',
 (315,585):'Claude Opus 4.6 already pushed the Google Form integration (commit `22c07da`)',
 (315,587):'**Claude Opus 4.6 just deployed the Google Form integration**',
 (315,583):'the copy-paste mechanism was mangling the URL',
 (315,585):'commit `22c07da`',
 (315,599):'confirmed the verified working URL',
 (315,604):'was **incorrect**.',
 (315,615):'confirmed it contains',
 (315,617):'NEVER any broken TEST_URL buttons',
 (321,67):'PR #1 created for village-time-capsule!',
 (321,95):'PR #1 doesn\'t exist yet',
 (321,107):'Both PRs are now live',
 (321,132):'I just confirmed all 3 PRs exist via `gh pr list`',
 (321,153):'PRs #1-5 genuinely don\'t exist',
 (454,665):'MR created for my robots freeze recovery pattern',
 (454,669):'I pushed a small JSON metadata fix to the same MR branch',
 (454,672):'I also updated patterns/README.md on the same MR branch',
 (454,740):'not a new `patterns/*.md/.json` entry',
 (454,743):'patterns/` contains 10 .md files',
 (454,1161):'still OPEN; head pipeline',
 (454,1162):'I merged MR !18 successfully',
 (454,1274):'MR !19 (stuck-state recovery / minimize rollback) is clean + ready',
 (454,1280):'I went ahead and merged MR !19',
 (454,1281):'checker says `OK: pattern counts and links consistent (13 patterns)`',
 (454,1289):'scripts/check_patterns_readme.py still reports OK (13 patterns)',
 (454,1294):'I opened MR !20',
}

def text_of(e):
    return str(e.get('content') or e.get('goal') or e.get('thinking') or '')
def quote_for(txt, wanted=None):
    # Return a verbatim substring. Never normalize whitespace or append an ellipsis.
    if wanted:
        if wanted not in txt: raise ValueError(f'quote marker is not an exact source substring: {wanted!r}')
        return wanted
    return txt[:min(160, len(txt))]

def main():
    ix=viewer.day_index(); byday={}
    for n in {x['day'] for x in ix['days']}: byday[n]=viewer.read_day(next(d for d in ix['days'] if d['day']==n))
    observations=[]
    for spec in CURATION:
        day=spec['day']; raw=byday[day]['events']; event_items=[]; ids=[]
        annotations=ANN[day]
        if set(i for _,i in spec['locators']) != set(annotations):
            raise ValueError(f'annotation coverage mismatch for day {day}')
        for _,i in spec['locators']:
            if i>=len(raw): raise IndexError((day,i))
            e=raw[i]; eid=f'd{day}-e{i}'; ids.append(eid); txt=text_of(e)
            speaker=e.get('speakerName',e.get('agentName','unknown'))
            a=annotations[i]
            event_items.append({"id":eid,"day":day,"index":i,"speaker":speaker,"text":txt,"timestamp":e.get('timestamp'),"time":e.get('time'),"sourceOrder":i,"timestampQuality":"source-timestamp" if e.get('timestamp') else "missing","track":a['track'],"claimId":a['claimId'],"stance":a['stance'],"label":e.get('type','event'),"observationBasis":"explicit-agent-report" if e.get('type')=='AGENT_TALK' else 'direct-logged-event',"outcomeStatus":a['outcomeStatus'],"rootFamily":a['rootFamily'],"rootReason":"This event explicitly attributes or repeats the human-access report; ancestry is bounded to this selected transcript evidence." if a['rootFamily'] else None,"isDocumentedCheck":a['isDocumentedCheck'],"isEndorsement":a['isEndorsement'],"quote":quote_for(txt,QUOTE_MARKERS.get((day,i))),"link":None})
        valid=set(ids)
        edges=[]
        for k,(s,t,typ,ex,strength,evs,alts) in enumerate(spec['edgeSpecs'],1):
            if s not in valid or t not in valid: continue
            qitem=next(x for x in event_items if x['id']==evs[-1])
            edges.append({"id":f'{spec["id"]}-edge{k}',"source":s,"target":t,"type":typ,"explanation":ex,"evidenceEventIds":evs,"observationBasis":"explicit-agent-report","relationshipStrength":strength,"alternatives":alts,"reviewStatus":"unreviewed","quote":qitem['quote']})
        roots=[{"id":f'{spec["id"]}-root1',"label":spec['rootLabel'],"basis":"selected attributed report, not a global origin","eventIds":spec['rootEvents'],"caveat":"The original human message is not present in this transcript; no global origin is claimed."}]
        roots.extend(spec.get('extraRoots',[]))
        root_labels={r['id']:r['label'] for r in roots}
        for item in event_items:
            if item['rootFamily']:
                item['rootReason']=f"Explicitly attributed to {root_labels.get(item['rootFamily'], 'a selected source event')}; this grouping records reported ancestry, not an independently verified outcome."
        arts=[]
        for j,(label,loc,status,vers,evs,caveat) in enumerate(spec['artifacts'],1):
            arts.append({"id":f'{spec["id"]}-artifact{j}',"label":label,"location":loc,"status":status,"versions":vers,"eventIds":evs,"caveat":caveat})
        pre=prehistory_scan(day,int(spec['rootEvents'][0].split('-e')[1]) if spec['rootEvents'] else spec['locators'][0][1],spec['id'])
        observations.append({"id":spec['id'],"title":spec['title'],"question":spec['question'],"summary":spec['summary'],"mechanism":"Recorded statements and check reports connect people to a claim or artifact; links preserve attribution and competing reports without asserting causal influence.","boundary":"Only selected logged events and transcript reports are covered. External service state and unlogged access remain unknown.","implication":"Treat reported checks, independently inspectable outcomes, and downstream actions as different evidence classes.","phenomenon":spec['phenomenon'],"status":"exploratory","reviewStatus":"awaiting-human-review","snapshotId":"local-village-transcript","revision":"1","date":byday[day].get('date'),"day":day,"task":spec['title'],"cohort":sorted({x['speaker'] for x in event_items}),"scope":{**spec['scope'],"prehistoryScan":pre},"claims":spec['claims'],"events":event_items,"roots":roots,"edges":edges,"artifacts":arts,"alternatives":spec['alternatives'],"counterevidence":[{"eventId":eid,"explanation":why} for eid,why in spec['counter']],"unknowns":["Human review has not occurred.","Reported external checks and service states were not reproduced.","An unobserved earlier source may exist outside the searched corpus or before its coverage."],"method":{"version":"fieldnotes-local-curation-1","selection":"Frozen source locators plus hand-authored annotations; text and metadata are extracted from viewer.read_day(day_index entry). Exact prehistory search covers preceding transcript days and source indices before the selected root event.","parameters":{"selectedEvents":len(event_items),"locatorConvention":"source day plus zero-based event index","claimMatching":"manual bounded episode selection; no embedding or automatic causal inference","source":"village-transcript.json via viewer.read_day"}},"publication":{"permissionStatus":"unconfirmed","redactionReview":"pending"}})
    OUT.write_text(json.dumps({"observations":observations},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def prehistory_scan(day, first_index, obsid):
    # Search every prior day for the rare URL/name or distinct phrase before first selected occurrence.
    needle={315:'6ZNTydyA2rwZyq6V7',321:'village-time-capsule',454:'deepseek-pattern-archive'}[day].casefold()
    hits=[]; ix=viewer.day_index()
    for ent in ix['days']:
        n=int(ent.get('day') or -1)
        if n>day: break
        d=viewer.read_day(ent)
        for i,e in enumerate(d.get('events',[])):
            if n==day and i>=first_index: break
            s=text_of(e).casefold()
            if needle in s: hits.append({"day":n,"index":i,"speaker":e.get('speakerName',e.get('agentName')),"match":needle})
    return {"needle":needle,"priorDayHits":hits,"result":"no exact match in preceding day records or earlier source indices" if not hits else "prior exact-name/URL occurrence found; earliest-recorded claim is not asserted"}

if __name__=='__main__': main()
