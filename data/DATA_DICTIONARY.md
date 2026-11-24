# IPL Auction Optimizer — Data Dictionary

**Last updated:** November 2026  
**Author:** Ranveer Singh

---

## Processed Datasets

### 1. `auction_unified.csv` / `.parquet`
**Description:** Unified IPL auction records from 2013–2025, combining auction history with player roles and origins.  
**Rows:** 1,279 | **Columns:** 8  
**Granularity:** One row per player per auction year

| Column | Type | Description | Example | Coverage |
|--------|------|-------------|---------|----------|
| `player` | string | Full player name (auction format) | Virat Kohli | 100% |
| `auction_year` | int | Year of the auction | 2022 | 100% |
| `team` | string | Franchise that purchased the player | Royal Challengers Bangalore | 100% |
| `base_price_cr` | float | Base/reserve price in crores (₹ Cr) | 2.00 | 100% |
| `sold_price_cr` | float | Final sold price in crores (₹ Cr) | 15.00 | 100% |
| `status` | string | Auction outcome: "sold" or "unsold" | sold | 100% |
| `role` | string | Player role: Batsman, Bowler, All-Rounder, Wicketkeeper | All-Rounder | 82% |
| `player_origin` | string | Nationality classification: Indian or Overseas | Indian | 82% |

**Notes:**
- Prices are in Indian crores (1 crore = 10 million INR)
- Role data sourced from IPL Auction Dashboard (2013–2022) and 2025 auction CSV
- ~18% of records (mainly 2023, some 2013) are missing role/origin data

---

### 2. `batting_season_stats.csv` / `.parquet`
**Description:** Season-level batting statistics aggregated from ball-by-ball data.  
**Rows:** 2,617 | **Columns:** 12  
**Granularity:** One row per batter per season

| Column | Type | Description | Formula | Example |
|--------|------|-------------|---------|---------|
| `batter` | string | Player name (cricsheet initials format) | — | V Kohli |
| `season` | int | IPL season year | — | 2023 |
| `matches` | int | Matches batted in that season | count(distinct match_id) | 14 |
| `total_runs` | int | Total runs scored | sum(batsman_runs) | 639 |
| `total_balls` | int | Total balls faced | count(deliveries) | 475 |
| `total_fours` | int | Number of fours hit | count(batsman_runs == 4) | 56 |
| `total_sixes` | int | Number of sixes hit | count(batsman_runs == 6) | 23 |
| `total_dismissals` | int | Number of times dismissed | sum(is_wicket) | 12 |
| `strike_rate` | float | Batting strike rate | (total_runs / total_balls) × 100 | 134.53 |
| `batting_avg` | float | Batting average | total_runs / total_dismissals | 53.25 |
| `boundary_pct` | float | Percentage of runs from boundaries | ((4s×4 + 6s×6) / total_runs) × 100 | 56.80 |

**Notes:**
- Computed from 260,920 ball-by-ball deliveries across 1,095 matches (2008–2024)
- `batting_avg` equals `total_runs` when `total_dismissals` = 0 (not out all innings)
- Players who only bowled and never batted do not appear in this dataset

---

### 3. `bowling_season_stats.csv` / `.parquet`
**Description:** Season-level bowling statistics aggregated from ball-by-ball data.  
**Rows:** 1,948 | **Columns:** 9  
**Granularity:** One row per bowler per season

| Column | Type | Description | Formula | Example |
|--------|------|-------------|---------|---------|
| `bowler` | string | Player name (cricsheet initials format) | — | JJ Bumrah |
| `season` | int | IPL season year | — | 2023 |
| `matches` | int | Matches bowled in that season | count(distinct match_id) | 16 |
| `total_runs_conceded` | int | Total runs given away | sum(total_runs) | 320 |
| `total_balls` | int | Total legal deliveries bowled | count(balls) | 384 |
| `total_wickets` | int | Wickets taken | sum(is_wicket) | 20 |
| `overs` | float | Overs bowled | total_balls / 6 | 64.0 |
| `economy` | float | Runs conceded per over | total_runs_conceded / overs | 5.00 |
| `bowling_avg` | float | Runs conceded per wicket | total_runs_conceded / total_wickets | 16.00 |

**Notes:**
- `bowling_avg` is NaN when `total_wickets` = 0 (bowled but took no wickets)
- `total_runs` includes extras (wides, no-balls, leg-byes, byes conceded)

---

### 4. `ml_auction_dataset.csv` / `.parquet`
**Description:** The ML-ready dataset — sold auction records enriched with 22 prior-season performance features.  
**Rows:** 1,279 | **Columns:** 31  
**Granularity:** One row per sold player per auction year  
**Target variable:** `sold_price_cr`

#### Auction Columns (9)

| Column | Type | Description |
|--------|------|-------------|
| `player` | string | Full player name (auction format) |
| `auction_year` | int | Year of auction |
| `team` | string | Purchasing franchise |
| `base_price_cr` | float | Base/reserve price (₹ Cr) |
| `sold_price_cr` | float | **TARGET** — Final sold price (₹ Cr) |
| `status` | string | Always "sold" in this dataset |
| `role` | string | Player role |
| `player_origin` | string | Indian or Overseas |
| `bbb_name` | string | Resolved ball-by-ball name for matching |

#### Batting Features (11)

| Column | Type | Description | Zero means |
|--------|------|-------------|------------|
| `career_runs` | int | Total IPL runs before this auction | No prior batting data |
| `career_balls_faced` | int | Total balls faced in IPL before auction | No prior batting data |
| `career_batting_sr` | float | Career strike rate before auction | No prior batting data |
| `career_batting_avg` | float | Mean of season batting averages | No prior batting data |
| `career_fours` | int | Total career fours | No prior batting data |
| `career_sixes` | int | Total career sixes | No prior batting data |
| `career_boundary_pct` | float | Mean of season boundary percentages | No prior batting data |
| `ipl_seasons_batted` | int | Number of IPL seasons batted before auction | No prior batting data |
| `career_bat_matches` | int | Total IPL matches batted in | No prior batting data |
| `recent_runs` | int | Runs in last 2 seasons before auction | No recent batting |
| `recent_batting_sr` | float | Strike rate in last 2 seasons | No recent batting |

#### Bowling Features (7)

| Column | Type | Description | Zero means |
|--------|------|-------------|------------|
| `career_wickets` | int | Total IPL wickets before auction | No prior bowling data |
| `career_bowl_balls` | int | Total balls bowled in IPL | No prior bowling data |
| `career_economy` | float | Career economy rate | No prior bowling data |
| `career_bowling_avg` | float | Mean of season bowling averages | No prior bowling data |
| `ipl_seasons_bowled` | int | Number of seasons bowled | No prior bowling data |
| `recent_wickets` | int | Wickets in last 2 seasons | No recent bowling |
| `recent_economy` | float | Economy in last 2 seasons | No recent bowling |

#### General Features (4)

| Column | Type | Description |
|--------|------|-------------|
| `total_ipl_seasons` | int | Max of seasons batted/bowled — overall IPL experience |
| `total_potm` | int | Career Player of the Match awards before auction |
| `price_ratio` | float | sold_price_cr / base_price_cr — auction premium multiplier |
| `has_ipl_experience` | int | Binary flag: 1 if any prior IPL stats exist, 0 otherwise |

**Important design decisions:**
- **No data leakage:** All features use only seasons *before* the auction year
- **"Recent form" window:** Last 2 IPL seasons before the auction — captures momentum
- **Zero vs NaN:** Features are 0 for uncapped players (no prior data), enabling ML models to learn the "uncapped player discount" pattern
- **`price_ratio`** can be NaN when `base_price_cr` is 0

---

## Raw Data Sources

| File | Source | Description |
|------|--------|-------------|
| `matches.csv` | avinashyadav16/ipl-analytics | Match-level IPL data 2008–2024 |
| `deliveries.csv` | avinashyadav16/ipl-analytics | Ball-by-ball data (⚠️ has trailing whitespace) |
| `auction_history.csv` | psai09599-stack/IPL-Auction-Intelligence | Auction records 2013–2025 |
| `player_id_map.csv` | psai09599-stack/IPL-Auction-Intelligence | 620 name mappings (auction → cricsheet) |
| `IPLPlayerAuctionData.csv` | sameekshasingh1809/ipl-auction-dashboard | Player roles & origins 2013–2022 |
| `auction_2025.csv` | psai09599-stack/IPL-Auction-Intelligence | 2025 auction with player types |
| `Player data.csv` | psai09599-stack/IPL-Auction-Intelligence | Player metadata (messy multi-row header) |

---

## Known Limitations

1. **Name matching at 84%** — 16% of sold players have zero prior IPL features because they are genuinely uncapped (first auction) or their names could not be resolved across datasets
2. **Role/origin coverage at 82%** — Missing for some 2013 and 2023 records
3. **Bowling wickets** include only bowler dismissals — run-outs where fielder ≠ bowler are not counted as wickets
4. **Season boundaries** — Ball-by-ball data runs 2008–2024; auction data 2013–2025. Players auctioned in 2025 have stats through 2024
5. **No domestic/international stats** — Features are IPL-only; a player's international career is not captured
