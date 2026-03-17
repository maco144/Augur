"""
Augur — Reference Class Playbook Library

Structured base-rate anchoring templates across 12+ domain categories for use
in reference class forecasting.  Each entry records a historical frequency or
probability, its source, time window, and caveats so that specialists can anchor
their probability estimates on empirical priors before adjusting for specifics.

Public helpers:
    get_base_rates(category)           — all templates in a category
    get_anchor(category, subcategory)  — single exact lookup
    list_categories()                  — available category names
    search_base_rates(query)           — keyword search across all entries
"""
from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Template schema (each dict)
# ---------------------------------------------------------------------------
#   category      : str   — top-level domain
#   subcategory   : str   — specific phenomenon
#   base_rate     : float — historical frequency / probability (0-1)
#   source        : str   — provenance of the estimate
#   time_period   : str   — historical window the rate covers
#   notes         : str   — caveats, conditions, limitations
#   last_updated  : str   — ISO date of last review
# ---------------------------------------------------------------------------


BASE_RATE_REGISTRY: list[dict] = [
    # -----------------------------------------------------------------------
    # GEOPOLITICS
    # -----------------------------------------------------------------------
    {
        "category": "geopolitics",
        "subcategory": "regime_change",
        "base_rate": 0.015,
        "source": "Polity IV dataset; Geddes, Wright & Frantz autocratic breakdown data 1946-2010",
        "time_period": "1946-2010",
        "notes": "Annual probability of an authoritarian regime collapsing or transitioning. Rate is higher for personalist regimes (~2.5%/yr) vs. single-party (~1%/yr).",
        "last_updated": "2025-06-01",
    },
    {
        "category": "geopolitics",
        "subcategory": "interstate_conflict_onset",
        "base_rate": 0.005,
        "source": "Correlates of War (COW) MID dataset; UCDP/PRIO Armed Conflict Dataset",
        "time_period": "1946-2023",
        "notes": "Annual probability of a new interstate armed conflict (≥25 battle deaths) for any given dyad of states. Rate for major-power dyads is lower (~0.1%/yr).",
        "last_updated": "2025-06-01",
    },
    {
        "category": "geopolitics",
        "subcategory": "sanctions_imposition",
        "base_rate": 0.03,
        "source": "Global Sanctions Database (GSDB); Hufbauer et al. 'Economic Sanctions Reconsidered'",
        "time_period": "1950-2023",
        "notes": "Annual probability that any given country is newly targeted by comprehensive international sanctions. Rate increased post-2014.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "geopolitics",
        "subcategory": "treaty_compliance",
        "base_rate": 0.75,
        "source": "Chayes & Chayes 'The New Sovereignty' (1995); compliance literature meta-analysis",
        "time_period": "1945-2020",
        "notes": "Approximate rate at which states substantially comply with international treaty obligations. Varies widely by treaty type; arms control ~65%, trade ~85%.",
        "last_updated": "2025-06-01",
    },
    # -----------------------------------------------------------------------
    # TECHNOLOGY
    # -----------------------------------------------------------------------
    {
        "category": "technology",
        "subcategory": "startup_success_rate",
        "base_rate": 0.10,
        "source": "CB Insights; Startup Genome Report; historical VC portfolio data",
        "time_period": "2005-2023",
        "notes": "~10% of VC-backed startups achieve a successful exit (IPO or acquisition above investment). ~75% fail to return capital. Seed-stage failure rate ~90%.",
        "last_updated": "2025-09-01",
    },
    {
        "category": "technology",
        "subcategory": "technology_mass_adoption",
        "base_rate": 0.25,
        "source": "Comin & Hobijn 'Cross-country technology adoption'; Our World in Data technology adoption curves",
        "time_period": "1900-2023",
        "notes": "Probability that a genuinely novel technology reaches 50% household/enterprise penetration within 15 years of commercial availability. Accelerating trend: internet took ~10yr, smartphones ~7yr.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "technology",
        "subcategory": "ai_capability_milestone_on_schedule",
        "base_rate": 0.35,
        "source": "AI Impacts survey of ML researchers; Metaculus AI benchmark resolution data",
        "time_period": "2015-2025",
        "notes": "Approximate. Fraction of AI capability milestones achieved by the median expert-predicted date. Experts tend to be overconfident on timelines; median predictions are ~2x too optimistic.",
        "last_updated": "2025-09-01",
    },
    {
        "category": "technology",
        "subcategory": "major_tech_company_antitrust_breakup",
        "base_rate": 0.03,
        "source": "Historical DOJ/FTC antitrust actions; AT&T 1984, Standard Oil 1911 as precedent",
        "time_period": "1950-2025",
        "notes": "Approximate annual probability that a major tech company faces a mandated structural breakup. Most antitrust cases result in behavioral remedies, not divestiture.",
        "last_updated": "2025-06-01",
    },
    # -----------------------------------------------------------------------
    # MARKETS / FINANCE
    # -----------------------------------------------------------------------
    {
        "category": "markets",
        "subcategory": "us_recession_annual",
        "base_rate": 0.15,
        "source": "NBER business cycle dates 1854-2024; ~33 recessions in ~170 years",
        "time_period": "1854-2024",
        "notes": "~15% annual probability of the US being in or entering recession. Post-WWII rate is lower (~13%). Average expansion lasts ~5 years, contraction ~11 months.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "markets",
        "subcategory": "fed_rate_cut_cycle",
        "base_rate": 0.20,
        "source": "Federal Reserve historical rate decisions; FRED data",
        "time_period": "1971-2024",
        "notes": "Annual probability of the Fed initiating a new easing cycle (first cut after a tightening or hold period). ~11 easing cycles in ~53 years.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "markets",
        "subcategory": "sp500_annual_decline",
        "base_rate": 0.26,
        "source": "S&P 500 total return data 1928-2024 (Shiller/Damodaran)",
        "time_period": "1928-2024",
        "notes": "S&P 500 posts a negative calendar-year total return ~26% of years. Declines >20% occur ~10% of years.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "markets",
        "subcategory": "ipo_first_year_underperformance",
        "base_rate": 0.60,
        "source": "Ritter 'Initial Public Offerings' long-run IPO performance studies",
        "time_period": "1980-2023",
        "notes": "~60% of IPOs underperform the market in their first year. Median IPO underperforms the index by ~5-10% in the first 12 months.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "markets",
        "subcategory": "sovereign_debt_default",
        "base_rate": 0.02,
        "source": "Reinhart & Rogoff 'This Time Is Different'; S&P sovereign default studies",
        "time_period": "1800-2023",
        "notes": "Annual probability of sovereign default for any given country. EM sovereigns ~3-5%/yr; advanced economies <0.5%/yr.",
        "last_updated": "2025-06-01",
    },
    # -----------------------------------------------------------------------
    # CLIMATE / ENVIRONMENT
    # -----------------------------------------------------------------------
    {
        "category": "climate",
        "subcategory": "extreme_weather_event_annual",
        "base_rate": 0.80,
        "source": "NOAA National Centers for Environmental Information; EM-DAT disaster database",
        "time_period": "1980-2024",
        "notes": "Annual probability of at least one billion-dollar weather/climate disaster in the US. Has been ~100% since 2010. Frequency increasing: ~18 events/year in 2020s vs. ~3/year in 1980s.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "climate",
        "subcategory": "climate_policy_implementation",
        "base_rate": 0.35,
        "source": "Climate Action Tracker; UNEP Emissions Gap Report analysis of NDC implementation",
        "time_period": "2015-2024",
        "notes": "Approximate. Fraction of Paris Agreement NDC commitments that countries are on track to fully implement. Most countries fall short of stated targets.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "climate",
        "subcategory": "annual_global_temperature_record",
        "base_rate": 0.18,
        "source": "NASA GISS; NOAA global temperature anomaly records 1880-2024",
        "time_period": "1880-2024",
        "notes": "Probability that any given year sets a new global average temperature record. Rate has increased sharply: ~50% of years in 2014-2024 set records.",
        "last_updated": "2025-06-01",
    },
    # -----------------------------------------------------------------------
    # HEALTHCARE / BIOTECH
    # -----------------------------------------------------------------------
    {
        "category": "healthcare",
        "subcategory": "clinical_trial_phase1_success",
        "base_rate": 0.52,
        "source": "BIO/QLS Advisors 'Clinical Development Success Rates 2011-2020'",
        "time_period": "2011-2020",
        "notes": "Probability of a drug advancing from Phase I to Phase II. Oncology is lower (~45%), rare disease higher (~65%).",
        "last_updated": "2025-06-01",
    },
    {
        "category": "healthcare",
        "subcategory": "clinical_trial_phase2_success",
        "base_rate": 0.29,
        "source": "BIO/QLS Advisors 'Clinical Development Success Rates 2011-2020'",
        "time_period": "2011-2020",
        "notes": "Probability of a drug advancing from Phase II to Phase III. This is the highest-attrition phase. Oncology ~25%, infectious disease ~35%.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "healthcare",
        "subcategory": "clinical_trial_phase3_success",
        "base_rate": 0.58,
        "source": "BIO/QLS Advisors 'Clinical Development Success Rates 2011-2020'",
        "time_period": "2011-2020",
        "notes": "Probability of a drug advancing from Phase III to NDA/BLA submission. Higher for drugs with Phase II biomarker endpoints.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "healthcare",
        "subcategory": "fda_approval_after_nda",
        "base_rate": 0.90,
        "source": "FDA CDER annual reports; BIO/QLS Advisors dataset",
        "time_period": "2011-2020",
        "notes": "Probability of FDA approval once an NDA/BLA is filed. Rejection is rare at this stage; most issues resolved via complete response letters.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "healthcare",
        "subcategory": "overall_clinical_success_rate",
        "base_rate": 0.08,
        "source": "BIO/QLS Advisors 'Clinical Development Success Rates 2011-2020'",
        "time_period": "2011-2020",
        "notes": "Overall probability of a Phase I drug reaching FDA approval. Oncology ~5%, infectious disease ~17%, rare disease ~15%.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "healthcare",
        "subcategory": "pandemic_emergence_annual",
        "base_rate": 0.03,
        "source": "Marani et al. 'Intensity and frequency of extreme novel epidemics' (PNAS 2021)",
        "time_period": "1600-2023",
        "notes": "Annual probability of a novel pandemic with COVID-19-level impact. Increasing trend due to zoonotic spillover frequency. ~2% historical, possibly 3-4% going forward.",
        "last_updated": "2025-06-01",
    },
    # -----------------------------------------------------------------------
    # ELECTIONS / POLITICS
    # -----------------------------------------------------------------------
    {
        "category": "elections",
        "subcategory": "us_presidential_incumbent_advantage",
        "base_rate": 0.67,
        "source": "US presidential election history 1900-2024; FairModel fundamentals",
        "time_period": "1900-2024",
        "notes": "Incumbent presidents (or incumbent-party candidates) win re-election ~67% of the time. Economy is the strongest predictor of deviation.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "elections",
        "subcategory": "us_polling_accuracy",
        "base_rate": 0.75,
        "source": "FiveThirtyEight/Silver Bulletin polling accuracy databases; AAPOR post-election analyses",
        "time_period": "1998-2024",
        "notes": "Probability that final polling average correctly identifies the winner in a US statewide or national race. Systematic polling errors of 2-4 points are common.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "elections",
        "subcategory": "us_major_legislation_passage",
        "base_rate": 0.04,
        "source": "GovTrack bill tracking data; Congressional Research Service reports",
        "time_period": "2001-2024",
        "notes": "Probability that a bill introduced in Congress becomes law. ~10,000-14,000 bills introduced per Congress; ~300-500 enacted. Major legislation (appropriations-level) has higher passage rate ~15-25%.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "elections",
        "subcategory": "us_midterm_president_party_loss",
        "base_rate": 0.89,
        "source": "US House midterm election results 1934-2022",
        "time_period": "1934-2022",
        "notes": "The president's party has lost House seats in ~89% of midterm elections (exceptions: 1998, 2002). Average loss is ~26 seats.",
        "last_updated": "2025-06-01",
    },
    # -----------------------------------------------------------------------
    # CYBERSECURITY
    # -----------------------------------------------------------------------
    {
        "category": "cybersecurity",
        "subcategory": "major_breach_annual_enterprise",
        "base_rate": 0.25,
        "source": "Verizon DBIR 2024; IBM Cost of a Data Breach Report 2024",
        "time_period": "2018-2024",
        "notes": "Approximate annual probability that any given large enterprise (>1000 employees) experiences a material data breach. Varies significantly by industry; healthcare and finance higher.",
        "last_updated": "2025-09-01",
    },
    {
        "category": "cybersecurity",
        "subcategory": "ransomware_payment_rate",
        "base_rate": 0.29,
        "source": "Coveware Quarterly Ransomware Reports; Chainalysis 2024 Crypto Crime Report",
        "time_period": "2020-2024",
        "notes": "Fraction of ransomware victims that pay the ransom. Declining trend: was ~70% in 2020, fell to ~29% by 2024. Payment does not guarantee data recovery.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "cybersecurity",
        "subcategory": "critical_vulnerability_patch_30d",
        "base_rate": 0.50,
        "source": "Qualys TruRisk Research; Kenna Security (Cisco) vulnerability intelligence reports",
        "time_period": "2019-2024",
        "notes": "Fraction of critical/high-severity CVEs patched within 30 days of disclosure across enterprise environments. Median time-to-patch for critical CVEs is ~30-35 days.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "cybersecurity",
        "subcategory": "zero_day_exploitation_annual",
        "base_rate": 0.70,
        "source": "Google Project Zero; Mandiant zero-day exploitation tracking",
        "time_period": "2019-2024",
        "notes": "Probability that at least one zero-day vulnerability is actively exploited in a major software platform (OS, browser, enterprise) in a given year. Effectively ~100% for the ecosystem as a whole; 70+ zero-days exploited in 2023.",
        "last_updated": "2025-06-01",
    },
    # -----------------------------------------------------------------------
    # ENERGY
    # -----------------------------------------------------------------------
    {
        "category": "energy",
        "subcategory": "renewable_capacity_target_met",
        "base_rate": 0.40,
        "source": "IEA World Energy Outlook; IRENA Renewable Capacity Statistics",
        "time_period": "2010-2024",
        "notes": "Approximate. Fraction of national renewable energy capacity targets met on schedule. Solar deployment has outpaced targets; wind and storage often lag.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "energy",
        "subcategory": "major_oil_supply_disruption_annual",
        "base_rate": 0.20,
        "source": "EIA 'Oil Supply Disruption Reference'; historical OPEC/geopolitical disruption data",
        "time_period": "1970-2024",
        "notes": "Annual probability of a supply disruption >1 million bbl/day. Includes geopolitical events (wars, sanctions, coups) and natural disasters.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "energy",
        "subcategory": "nuclear_plant_approval_to_operation",
        "base_rate": 0.50,
        "source": "IAEA PRIS database; World Nuclear Association construction timelines",
        "time_period": "2000-2024",
        "notes": "Approximate. Fraction of approved/planned nuclear plants that achieve commercial operation within announced timelines. Most experience significant delays (median delay ~5 years).",
        "last_updated": "2025-06-01",
    },
    # -----------------------------------------------------------------------
    # LEGAL / REGULATORY
    # -----------------------------------------------------------------------
    {
        "category": "legal",
        "subcategory": "us_antitrust_challenge_success",
        "base_rate": 0.35,
        "source": "DOJ Antitrust Division and FTC case outcome data; academic merger challenge studies",
        "time_period": "2000-2024",
        "notes": "Fraction of DOJ/FTC merger challenges that result in the deal being blocked or abandoned. Many challenged mergers settle with divestitures.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "legal",
        "subcategory": "major_regulation_implementation_on_time",
        "base_rate": 0.30,
        "source": "Government Accountability Office (GAO) regulatory implementation reports",
        "time_period": "2000-2024",
        "notes": "Approximate. Fraction of major federal regulations implemented by their originally stated effective date. Most are delayed by litigation, comment periods, or administration changes.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "legal",
        "subcategory": "scotus_case_reversal_rate",
        "base_rate": 0.70,
        "source": "SCOTUSblog statistics; Supreme Court Database",
        "time_period": "2000-2024",
        "notes": "Probability that the Supreme Court reverses or vacates the lower court decision in cases it agrees to hear. The Court selects cases where it believes the lower court erred.",
        "last_updated": "2025-06-01",
    },
    # -----------------------------------------------------------------------
    # SPACE / AEROSPACE
    # -----------------------------------------------------------------------
    {
        "category": "space",
        "subcategory": "orbital_launch_success",
        "base_rate": 0.95,
        "source": "Space Launch Report; Jonathan McDowell launch logs",
        "time_period": "2018-2024",
        "notes": "Overall orbital launch success rate across all providers. Mature vehicles (Falcon 9, Soyuz) exceed 98%. New vehicles have ~85-90% success on early flights.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "space",
        "subcategory": "space_mission_schedule_slip",
        "base_rate": 0.80,
        "source": "NASA OIG audit reports; GAO space program assessments",
        "time_period": "2000-2024",
        "notes": "Fraction of major space missions (NASA, ESA) that experience significant schedule delays (>12 months). James Webb: 14 years late. Average major mission delay is ~3 years.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "space",
        "subcategory": "mars_mission_success",
        "base_rate": 0.50,
        "source": "The Planetary Society Mars mission catalog; NASA/ESA mission histories",
        "time_period": "1960-2024",
        "notes": "Overall success rate for Mars missions (orbit or landing). Has improved: post-2000 success rate is ~75%. Early Soviet missions had very high failure rates.",
        "last_updated": "2025-06-01",
    },
    # -----------------------------------------------------------------------
    # ECONOMICS
    # -----------------------------------------------------------------------
    {
        "category": "economics",
        "subcategory": "inflation_above_3pct_persistence",
        "base_rate": 0.65,
        "source": "Federal Reserve; BLS CPI data 1960-2024",
        "time_period": "1960-2024",
        "notes": "Conditional probability: given US CPI inflation is >3% in year T, the probability it remains >3% in year T+1. Inflation tends to be sticky once elevated.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "economics",
        "subcategory": "employment_recovery_to_peak",
        "base_rate": 0.70,
        "source": "BLS employment data; NBER business cycle analysis",
        "time_period": "1948-2024",
        "notes": "Probability that total US employment recovers to pre-recession peak within 24 months of the recession's end. Post-GFC recovery took 76 months; COVID recovery took 29 months.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "economics",
        "subcategory": "emerging_market_currency_crisis",
        "base_rate": 0.05,
        "source": "Laeven & Valencia 'Systemic Banking Crises Revisited' (IMF); Reinhart & Rogoff",
        "time_period": "1970-2023",
        "notes": "Annual probability that any given emerging-market country experiences a currency crisis (>15% depreciation + reserves depletion). Clustering effect: crises tend to be contagious.",
        "last_updated": "2025-06-01",
    },
    {
        "category": "economics",
        "subcategory": "us_gdp_growth_below_zero",
        "base_rate": 0.17,
        "source": "BEA GDP data; NBER recession dates 1948-2024",
        "time_period": "1948-2024",
        "notes": "Probability that annual US real GDP growth is negative. ~13 negative-growth years out of ~76. Single negative quarters are more common (~22% of quarters).",
        "last_updated": "2025-06-01",
    },
    # -----------------------------------------------------------------------
    # AI / ML
    # -----------------------------------------------------------------------
    {
        "category": "ai_ml",
        "subcategory": "frontier_model_capability_jump",
        "base_rate": 0.40,
        "source": "Epoch AI; ML benchmark progression tracking; historical model release data",
        "time_period": "2019-2025",
        "notes": "Approximate. Annual probability that a newly released frontier model shows a >10% improvement on major benchmarks (MMLU, HumanEval, etc.) over the previous SOTA. Rate has been high during the scaling era but may slow.",
        "last_updated": "2025-09-01",
    },
    {
        "category": "ai_ml",
        "subcategory": "ai_deployment_timeline_met",
        "base_rate": 0.30,
        "source": "Industry reports; Gartner Hype Cycle analysis; enterprise AI adoption surveys",
        "time_period": "2018-2025",
        "notes": "Approximate. Fraction of enterprise AI deployment projects completed on original timeline and budget. Most AI projects experience scope reduction or delay.",
        "last_updated": "2025-09-01",
    },
    {
        "category": "ai_ml",
        "subcategory": "ai_safety_incident_rate",
        "base_rate": 0.15,
        "source": "AIAAIC Repository; AI Incident Database (AIID); media tracking",
        "time_period": "2020-2025",
        "notes": "Approximate. Annual probability that a major AI system (>1M users) causes a widely-reported safety incident (harmful output, bias event, privacy breach). Rate is increasing with deployment scale.",
        "last_updated": "2025-09-01",
    },
    {
        "category": "ai_ml",
        "subcategory": "ml_paper_reproducibility",
        "base_rate": 0.50,
        "source": "Raff 'A Step Toward Quantifying Independently Reproducible ML Research' (NeurIPS 2019); ML Reproducibility Challenge results",
        "time_period": "2015-2024",
        "notes": "Fraction of ML research papers whose main results can be independently reproduced. Varies by subfield; reinforcement learning is lower (~35%), supervised learning higher (~60%).",
        "last_updated": "2025-06-01",
    },
    {
        "category": "ai_ml",
        "subcategory": "ai_regulation_passage_annual",
        "base_rate": 0.25,
        "source": "OECD AI Policy Observatory; national AI legislation tracking",
        "time_period": "2020-2025",
        "notes": "Approximate. Annual probability that a given OECD country enacts significant new AI-specific legislation. EU AI Act passed 2024; US federal AI regulation still pending as of 2025.",
        "last_updated": "2025-09-01",
    },
]


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------

def list_categories() -> list[str]:
    """Return sorted list of unique category names in the registry."""
    return sorted({entry["category"] for entry in BASE_RATE_REGISTRY})


def get_base_rates(category: str) -> list[dict]:
    """Return all base-rate templates in *category* (case-insensitive)."""
    cat = category.strip().lower()
    return [entry for entry in BASE_RATE_REGISTRY if entry["category"] == cat]


def get_anchor(category: str, subcategory: str) -> Optional[dict]:
    """Exact lookup by category + subcategory.  Returns None if not found."""
    cat = category.strip().lower()
    sub = subcategory.strip().lower()
    for entry in BASE_RATE_REGISTRY:
        if entry["category"] == cat and entry["subcategory"] == sub:
            return entry
    return None


def search_base_rates(query: str) -> list[dict]:
    """Keyword search across category, subcategory, notes, and source fields."""
    tokens = query.strip().lower().split()
    if not tokens:
        return []
    results: list[dict] = []
    for entry in BASE_RATE_REGISTRY:
        searchable = " ".join([
            entry["category"],
            entry["subcategory"],
            entry["notes"],
            entry["source"],
        ]).lower()
        if all(tok in searchable for tok in tokens):
            results.append(entry)
    return results
