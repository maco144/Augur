# Citation & Claims Audit — March 2025

**Audit date:** 2025-03-17
**Scope:** UC Berkeley AI Alignment January 2025 research citations; Polymarket Q4 2024 TVL claims
**Purpose:** Legal substantiation check before prospect outreach

---

## 1. UC Berkeley AI Alignment Research — January 2025

### What can be verified

**Emmons (2025) — "The Alignment Problem Under Partial Observability"**
- UC Berkeley EECS Technical Report EECS-2025-1, January 2025.
- Author: Scott Emmons. Advisor: Stuart J. Russell (CHAI founder).
- A doctoral thesis using game-theoretic assistance-game frameworks to study alignment under partial observability, including RLHF failure modes and the off-switch problem.
- Source: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-1.html

**"Bidirectional Human-AI Alignment: Emerging Challenges and Opportunities"**
- Co-authored by Marti Hearst (UC Berkeley School of Information) and others.
- Published as a CHI 2025 Extended Abstract (CHI EA '25) and associated ICLR 2025 Workshop.
- Proposes a bidirectional alignment framework based on a survey of 400+ alignment papers.
- Note: While Berkeley-affiliated, the paper is a multi-institution collaboration and was published for a 2025 conference (likely submitted/circulated late 2024 or early 2025). It does NOT specifically address ensemble forecasting, calibration, or multi-agent prediction systems.
- Source: https://www.ischool.berkeley.edu/research/publications/2025/bidirectional-human-ai-alignment-emerging-challenges-and-opportunities

**CHAI general research directions**
- CHAI's stated research areas include multi-agent perspectives, value alignment, and models of bounded rationality (https://humancompatible.ai/).
- These topics are adjacent to ensemble forecasting concepts but CHAI has not published work specifically about ensemble forecasting calibration or multi-model prediction approaches in the January 2025 timeframe.

### What CANNOT be verified

- **Any claim that Berkeley/CHAI published research specifically endorsing ensemble forecasting, multi-specialist prediction, or calibration methods in January 2025.** No such paper was found.
- **Any claim that Berkeley research validates Augur's specific architecture.** The Emmons thesis addresses alignment theory, not forecasting system design.
- **Any claim linking Berkeley AI alignment findings to prediction-market accuracy or multi-agent forecasting superiority.** This connection does not appear in the literature.

### Recommended safe phrasing

> "Recent UC Berkeley AI alignment research (Emmons, 2025) highlights the importance of robust multi-agent frameworks and the challenges of partial observability in AI systems — themes that motivate ensemble approaches to forecasting."

This phrasing is defensible because:
- The Emmons thesis does use multi-agent game-theoretic frameworks.
- Partial observability is a real challenge that ensemble methods address.
- The language says "motivate" rather than "validate" or "confirm."

### Claims to remove or correct

| Problematic claim pattern | Issue | Recommendation |
|---|---|---|
| "Berkeley research proves ensemble forecasting is superior" | No such finding exists | Remove entirely |
| "CHAI January 2025 paper on multi-agent calibration" | No such paper exists | Remove or replace with accurate Emmons (2025) citation |
| "UC Berkeley validates our approach" | Implies endorsement | Rewrite as thematic alignment, not endorsement |
| "Leading AI alignment researchers at Berkeley recommend multi-specialist architectures" | Unsubstantiable | Remove |

---

## 2. Polymarket Q4 2024 TVL & Volume Claims

### What can be verified

**Cumulative 2024 trading volume: ~$9 billion**
- Widely reported by The Block, ChainCatcher, and others.
- However, an important caveat: Polymarket was accused of double-counting volume due to smart contract events emitting separate OrderFilled events for maker and taker sides. Corrected monthly volumes for Oct-Nov 2024 may be roughly half (~$1.25B/month each) of the ~$2.5B figures on public dashboards.
- Sources: The Block (https://www.theblock.co/post/333050), Yahoo Finance double-counting report.

**November 2024 monthly volume: reported at $2.63 billion (all-time high)**
- Widely cited figure, but subject to the double-counting caveat above.
- Corrected figure may be closer to $1.25-1.3 billion.

**Peak open interest: ~$510 million (November 2024)**
- Open interest hit an all-time high around the U.S. election.

**TVL peak: reported figures vary**
- The Defiant reported Polymarket topping $330 million TVL.
- Other sources cite $250 million TVL peak in Q4 2024.
- One source (Bitget) cited $4 billion "total value locked" but this figure appears to conflate cumulative volume with TVL and is unreliable.
- Safe range: TVL peaked between $250M and $500M in Q4 2024, depending on the metric and source.

**Active traders: 314,500 (December 2024 peak)**
- Reported by The Block.

**Election market dominance: ~65% of Q4 volume**
- U.S. Presidential Election markets accounted for approximately $2.8 billion of Q4 volume (before double-counting correction).

**Post-election decline: 34-84% drop in December**
- Multiple sources confirm steep volume decline after election settlement. DL News reported an 84% plummet; The Block reported a 34% decline month-over-month.

### What CANNOT be verified

- **Any single precise TVL figure for Q4 2024.** Sources disagree significantly ($250M to $500M range). The "$4 billion TVL" figure from one source is almost certainly cumulative volume, not TVL.
- **Uncorrected volume figures as fact.** The double-counting issue means raw dashboard numbers may overstate volume by ~2x.

### Recommended safe phrasing

> "Polymarket processed over $9 billion in cumulative trading volume in 2024, with Q4 driven by the U.S. presidential election. Open interest peaked at approximately $510 million in November, and the platform reached over 300,000 active traders by year-end."

This phrasing is defensible because:
- Cumulative $9B figure is widely reported and less affected by the double-counting dispute (which concerns monthly breakdowns).
- Open interest of ~$510M is a separately tracked on-chain metric.
- 300,000+ active traders is conservative relative to the 314,500 reported figure.
- It avoids a specific TVL number, which is the most disputed metric.

### Claims to remove or correct

| Problematic claim pattern | Issue | Recommendation |
|---|---|---|
| "Polymarket had $X billion TVL in Q4 2024" (where X > 1) | Likely conflates volume with TVL | Replace with open interest or cumulative volume figure |
| "$2.6B monthly volume in November 2024" stated as fact | Subject to double-counting dispute | Add qualifier: "as reported on public dashboards" or use "over $1 billion" |
| "Polymarket TVL of $4 billion" | Almost certainly wrong; conflates metrics | Remove |
| "Prediction markets are a $9 billion industry" | Conflates one platform's annual volume with market size | Rewrite to specify Polymarket's cumulative 2024 volume |

---

## 3. Summary & Outreach Recommendations

### Overall risk assessment

| Claim category | Risk level | Action |
|---|---|---|
| Berkeley AI alignment — thematic connection to ensemble methods | LOW | Permissible with careful phrasing |
| Berkeley AI alignment — specific endorsement of Augur | HIGH | Remove; no evidence exists |
| Polymarket volume — $9B cumulative 2024 | LOW | Safe to cite with source attribution |
| Polymarket TVL — specific Q4 figure | MEDIUM | Use open interest (~$510M) or range; avoid single TVL number |
| Polymarket active traders — 300K+ | LOW | Safe to cite |
| Polymarket — post-election decline narrative | LOW | Factual and well-documented |

### Pre-outreach checklist

- [ ] Remove any claims of Berkeley/CHAI endorsement of ensemble forecasting
- [ ] Replace specific Berkeley citation with Emmons (2025) thesis if a citation is needed
- [ ] Use "open interest" rather than "TVL" for Polymarket peak figures
- [ ] Add double-counting caveat or use conservative volume figures
- [ ] Ensure all numerical claims have source attribution

---

## Sources consulted

- [UC Berkeley EECS Technical Reports — Emmons 2025](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-1.html)
- [UC Berkeley School of Information — Bidirectional Human-AI Alignment](https://www.ischool.berkeley.edu/research/publications/2025/bidirectional-human-ai-alignment-emerging-challenges-and-opportunities)
- [CHAI — Center for Human-Compatible AI](https://humancompatible.ai/)
- [BAIR — Berkeley AI Research](https://bair.berkeley.edu/)
- [The Block — Polymarket's $9B Year](https://www.theblock.co/post/333050/polymarkets-huge-year-9-billion-in-volume-and-314000-active-traders-redefine-prediction-markets)
- [The Defiant — Prediction Markets TVL High](https://thedefiant.io/news/nfts-and-web3/crypto-prediction-markets-hit-tvl-all-time-high)
- [DL News — Polymarket Post-Election Volume Decline](https://www.dlnews.com/articles/markets/polymarket-volumes-and-users-plummet-after-trump-wins-election/)
- [Yahoo Finance — Polymarket Double-Counting Accusation](https://finance.yahoo.com/news/polymarket-accused-double-counting-trading-090924852.html)
- [World Metrics — Polymarket Statistics 2026](https://worldmetrics.org/polymarket-statistics/)
- [ChainCatcher — Polymarket 2025 Report](https://www.chaincatcher.com/en/article/2233047)
- [Token Terminal — Polymarket TVL](https://tokenterminal.com/explorer/projects/polymarket/metrics/tvl)
- [Polymarket Wikipedia](https://en.wikipedia.org/wiki/Polymarket)
