# Supplemental reference check

**Result: PASS with one optional date clarification.** All four papers listed in `build_deep_report.py` lines 104–107 are real. Their titles, lead-author labels, journal names and displayed years match current canonical metadata. No required correction was found.

This was an independent fresh-context check. The reviewer actually ran as `gpt-5.6-sol` with reasoning effort `xhigh` and `fork_turns=none`; no `mcp__codex` tool was called. Earlier project audit results were not used as evidence. The report script was read but not executed, and neither it nor the 56-entry citation ledger was modified.

## Findings

| Source line | Result | Canonical identity | Publication state |
|---:|---|---|---|
| 104 | KEEP | Cunyu Yuan, Xia Zhang, Luliang Tang, Danning Xu & Hong Yang. “Depression risk associated with idealized sentiment toward urban places on social media.” *Nature Cities* **3**, 916–928 (2026). DOI: [10.1038/s44284-026-00516-x](https://doi.org/10.1038/s44284-026-00516-x) | Online 9 September 2026; issue 10, October 2026. The supplied PDF still contains `Published online: xx xx xxxx`, but the current publisher metadata is final. |
| 105 | KEEP | Jiayu Wu, Running Chen, Zhizheng Cai, Yu Zhang, Corey T. Callaghan, Frank A. La Sorte & Baojing Gu. “Urban greening strategies for mitigating heat exposure and enhancing bird diversity.” *Nature Communications* **17**, 10619 (2026). DOI: [10.1038/s41467-026-77596-9](https://doi.org/10.1038/s41467-026-77596-9) | Online 8 September 2026; volume 17, issue 1, article 10619. The supplied PDF was an “Article in Press” proof; current publisher metadata is final. |
| 106 | KEEP | Jiayu Wu, Jidian Ruan, Wenjie Di, Jiayao Ying, Jiali Zhou, Zeyu Luo, Igor Rudan & Peige Song. “The global burden of hypertension preventable by urban greenness.” *Nature Health* **1**, 582–594 (2026). DOI: [10.1038/s44360-026-00090-5](https://doi.org/10.1038/s44360-026-00090-5) | Online 26 March 2026; issue 6, June 2026. |
| 107 | KEEP; optional clarification | Youlong Gu, Xiucheng Liang, Zicheng Fan, Yujun Hou & Filip Biljecki. “Street view video for urban sensing: Potential for measuring urban vitality.” *Computers, Environment and Urban Systems* **131**, 102509 (2027). DOI: [10.1016/j.compenvurbsys.2026.102509](https://doi.org/10.1016/j.compenvurbsys.2026.102509) | **Available online 17 September 2026; assigned to volume 131, January 2027.** The script’s `(2027)` is the correct issue/citation year. |

## Gu paper date determination

The apparent 2026/2027 conflict is a normal online-first/future-issue split:

- The supplied paper states `Computers, Environment and Urban Systems 131 (2027) 102509` and `Available online 17 September 2026`.
- [Crossref](https://api.crossref.org/works/10.1016%2Fj.compenvurbsys.2026.102509) reports volume 131, article 102509, with `published-print` and `issued` set to January 2027; its record was created on 17 September 2026.
- The [Elsevier Article Retrieval API](https://api.elsevier.com/content/article/PII:S0198971526001110?httpAccept=text/xml) returns the exact DOI, title and journal, with cover date `2027-01-31` and display date `January 2027`.
- The DOI resolved successfully to Elsevier PII `S0198971526001110`. A direct automated request to the ScienceDirect page returned HTTP 403, which is access control rather than evidence against the work. The exact ScienceDirect page and corresponding-author laboratory page were also independently indexed.

Therefore, `(2027)` should not be changed to `(2026)` in a conventional reference. If the table should make its chronology explicit, this optional replacement is precise:

**Old**

> Gu et al., Computers, Environment and Urban Systems (2027). Street view video for urban sensing: Potential for measuring urban vitality.

**New (optional)**

> Gu et al., Computers, Environment and Urban Systems 131, 102509 (2027; published online 17 September 2026). Street view video for urban sensing: Potential for measuring urban vitality.

## Evidence trail

The Nature DOI records and publisher HTML citation metadata agreed on titles, ordered authors, journals and final bibliographic fields:

- [Yuan et al., Nature Cities](https://www.nature.com/articles/s44284-026-00516-x): HTTP 200; publisher metadata gives online date 2026-09-09, volume 3, issue 10, pages 916–928.
- [Wu et al., Nature Communications](https://www.nature.com/articles/s41467-026-77596-9): HTTP 200; publisher metadata gives online date 2026-09-08, volume 17, issue 1, article 10619.
- [Wu et al., Nature Health](https://www.nature.com/articles/s44360-026-00090-5): HTTP 200; publisher metadata gives online date 2026-03-26, volume 1, issue 6, pages 582–594.
- [Gu et al., Urban Analytics Lab](https://ual.sg/publication/2027-ceus-svv/): HTTP 200; exact title, five-author order, journal and September 2026 posting date agree with the supplied article and registry record.

User-supplied evidence was read from `qa/paper_references/paper_1.txt` through `paper_4.txt`. Their SHA-256 hashes and per-entry evidence details are recorded in `SUPPLEMENTAL_REFERENCE_CHECK.json`. The short table uses these works for visual/evidence architecture and explicitly states that their data, causal assumptions, effect sizes and significance labels are not transferred; no contextual misattribution was found.
