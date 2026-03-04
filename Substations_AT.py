"""
Substations_AT.py

Central Austrian substation registry used by the grid preprocessing pipeline.

Purpose
-------
1) Provide a manual lookup for Austrian substations appearing in Lines_AT_and_Tielines_Parameters.csv.
2) Attach geographic and structural metadata to each substation:
      - lat/lon (WGS84)
      - federal state ("state")
3) Define Austrian Generation Shift Keys (GSK) on federal-state level.

Design Principles
-----------------
- Keys must match the canonical substation names produced by
  normalize_substation_name() in Lines_AT_and_Tielines_Parameters.py.
- State labels must match the keys used in GSK_STATE_TECH_SHARES.
- GSK shares are defined per federal state and generation group.
- For each generation group, state shares should sum to 1.0.
- If multiple substations belong to the same state, generation is
  distributed equally among substations of that state unless defined otherwise.

This file is intentionally manual and acts as the Austrian structural backbone
for country-to-substation disaggregation.
"""


# ============================================================
# COORDINATES LOOKUP
# ============================================================

SUBSTATION_LOOKUP = {

    "BISAMBERG": {
        "lat": 48.357501,
        "lon": 16.378611,
        "state": "NIEDEROESTERREICH"
    },
    "DREIBEIN MELLACH": {
        "lat": 46.910058,
        "lon": 15.491131,
        "state": "STEIERMARK"
    },
    "DUERNROHR": {
        "lat": 48.329166,
        "lon": 15.882501,
        "state": "NIEDEROESTERREICH"      
    },
    "ERNSTHOFEN": {
        "lat": 48.125555,
        "lon": 14.473611,
        "state": "NIEDEROESTERREICH"
    },
    "ETZERSDORF": {
        "lat": 48.268611,
        "lon": 15.736666,
        "state": "STEIERMARK"
    },
    "FEISTRITZ": {
        "lat": 46.535903,
        "lon": 14.172617,
        "state": "KAERNTEN"
    },
    "GREUTH": {
        "lat": 46.543172,
        "lon": 13.679811,
        "state": "NIEDEROESTERREICH"
    },
    "HAUSRUCK": {
        "lat": 48.103888,
        "lon": 13.899444,
        "state": "OBEROESTERREICH"
    },
    "HESSENBERG": {
        "lat": 47.396666,
        "lon": 15.020555,
        "state": "STEIERMARK"
    },
    "JOCHENSTEIN": {
        "lat": 48.519614,
        "lon": 13.710731,
        "state": "OBEROESTERREICH"
    },
    "KAINACHTAL": {
        "lat": 46.903611,
        "lon": 15.423055,
        "state": "STEIERMARK"
    },
    "KAPRUN": {
        "lat": 47.259166,
        "lon": 12.742222,
        "state": "SALZBURG"
    },
    "KLAUS": {
        "lat": 47.821111,
        "lon": 14.163055,
        "state": "OBEROESTERREICH"
    },
    "KLEDERING": {
        "lat": 48.139167,
        "lon": 16.432724,
        "state": "WIEN"
    },
    "KRONSTORF": {
        "lat": 48.11634980943017, 
        "lon": 14.469291234538055,
        "state": "OBEROESTERREICH"
    },
    "LIENZ": {
        "lat": 46.824722,
        "lon": 12.805277,
        "state": "KAERNTEN"
    },
    "MATREI": {
        "lat": 46.97900447281944, 
        "lon": 12.547447986041382,
        "state": "TIROL"
    },
    "MAYRHOFEN": {
        "lat": 47.162501,
        "lon": 11.850555,
        "state": "TIROL"
    },
    "MEININGEN": {
        "lat": 47.312222,
        "lon": 9.591667,
        "state": "VORARLBERG"
    },
    "NAUDERS": {
        "lat": 46.858279364679035, 
        "lon": 10.507109621484576,
        "state": "TIROL"
    },
    "NEUSIEDL": {
        "lat": 47.960277,
        "lon": 16.837777,
        "state": "BURGENLAND"
    },
    "OBERSIELACH": {
        "lat": 46.668333,
        "lon": 14.681388,
        "state": "KAERNTEN"
    },
    "OSTSTEIERMARK": {
        "lat": 47.091008,
        "lon": 15.725867,
        "state": "STEIERMARK"
    },
    "PHYRN": {
        "lat": 47.678611,
        "lon": 14.330833,
        "state": "OBEROESTERREICH"
    },
    "PONGAU": {
        "lat": 47.378333,
        "lon": 13.210277,
        "state": "SALZBURG"
    },
    "ROSEGG": {
        "lat": 46.568392,
        "lon": 14.027861,
        "state": "KAERNTEN"
    },
    "SALZACH": {
        "lat": 47.830957,
        "lon": 13.029631,
        "state": "SALZBURG"
    },
    "SALZBURG": {
        "lat": 47.888062,
        "lon": 13.067685,
        "state": "SALZBURG"
    },
    "SARASDORF": {
        "lat": 48.031985,
        "lon": 16.696037,
        "state": "NIEDEROESTERREICH"
    },
    "SATTLEDT": {
        "lat": 48.083384,
        "lon": 14.052179,
        "state": "OBEROESTERREICH"
    },
    "SCHAERDING": {
        "lat": 48.434512,
        "lon": 13.443211,
        "state": "OBEROESTERREICH"
    },
    "SCHWARZENBACH": {
        "lat": 47.27847,
        "lon": 12.58338,
        "state": "SALZBURG"
    },
    "ST PETER": {
        "lat": 48.256111,
        "lon": 13.080833,
        "state": "OBEROESTERREICH"
    },
    "SUEDBURGENLAND": {
        "lat": 47.250555,
        "lon": 16.276111,
        "state": "BURGENLAND"
    },
    "SW WEIBERN": {
        "lat": 48.17383430566289, 
        "lon": 13.70343698916205,
        "state": "OBEROESTERREICH"
    },
    "TAUERN": {
        "lat": 47.277777,
        "lon": 12.739722,
        "state": "SALZBURG"
    },
    "TERNITZ": {
        "lat": 47.707222,
        "lon": 16.064444,
        "state": "NIEDEROESTERREICH"
    },
    "VILL": {
        "lat": 47.236178, 
        "lon": 11.391747,
        "state": "TIROL"
    },
    "VILLACH SUED": {
        "lat": 46.57339747273104, 
        "lon": 13.863558625412924,
        "state": "KAERNTEN"
    },
    "WAGENHAM": {
        "lat": 48.09854491462521, 
        "lon": 13.104532652068805,
        "state": "OBEROESTERREICH"
    },
    "WALLSEE": {
        "lat": 48.164194,
        "lon": 14.69543,
        "state": "OBEROESTERREICH"
    },
    "WALGAUWERK": {
        "lat": 47.198879,
        "lon": 9.669674,
        "state": "VORARLBERG"
    },
    "WEISSENBACH": {
        "lat": 47.573888,
        "lon": 14.208055,
        "state": "STEIERMARK"
    },
    "WERBEN": {
        "lat": 47.432501,
        "lon": 9.716944,
        "state": "STEIERMARK"
    },
    "WESTTIROL": {
        "lat": 47.243034499433854, 
        "lon": 10.872802956196402,
        "state": "TIROL"
    },
    "WIEN SUEDOST": {
        "lat": 48.121944,
        "lon": 16.418055,
        "state": "WIEN"
    },
    "YBBSFELD": {
        "lat": 48.145833,
        "lon": 15.048055,
        "state": "NIEDEROESTERREICH"
    },
    "ZAYA": {
        "lat": 48.60591073097857, 
        "lon": 16.80049659447515,
        "state": "NIEDEROESTERREICH"
    },
    "ZELL AM ZILLER": {
        "lat": 47.233333,
        "lon": 11.898055,
        "state": "TIROL"
    },
    "ZELTWEG": {
        "lat": 47.181343,
        "lon": 14.726066,
        "state": "STEIERMARK"
    },
    "ZURNDORF": {
        "lat": 47.938288,
        "lon": 16.994841,
        "state": "BURGENLAND"
    },
}

# ============================================================
# GENERATION SHIFT KEYS (GSK)
# ============================================================


GSK_STATE_TECH_SHARES = {

    "BURGENLAND": {
        "gas": 0.0039, #Source: ÖNIP
        "coal": 0.0, #Source: ÖNIP
        "other_non_res": 0.0, #Source: ÖNIP
        "wind": 0.3160, #Source: ÖNIP
        "pv": 0.169166666666667, #Source: ÖNIP
        "other_res": 0.0394218134034166, #Source: ÖNIP
        "batteries": 0.0142857142857143, #Source: ÖNIP
        "ror": 0.00611572839893367, #Source: ÖNIP
        "hydro_turbine": 0.0, #Source: ÖNIP
        "demand": 0.0171507260891337, #Source: Data Christoph
        "hs_inflow": 0.0,  #Source: Data Christoph --> This is wrong! Use ÖNIP-Source
        "phs_inflow": 0.0,  #Source: Data Christoph  --> This is wrong! Use ÖNIP-Source
    },

    "KAERNTEN": {
        "gas": 0.0033,
        "coal": 0.0,
        "other_non_res": 0.0568047337278106,
        "wind": 0.0466666666666667,
        "pv": 0.088,
        "other_res": 0.173455978975033,
        "batteries": 0.157142857142857,
        "ror": 0.146307040928336,
        "hydro_turbine": 0.360018403496664,
        "demand": 0.0784301452178267,
        "hs_inflow": 0.360018403496664, #0.0115350713935864,
        "phs_inflow": 0.360018403496664, #0.654629065381002,
    },

    "NIEDEROESTERREICH": {
        "gas": 0.0202,
        "coal": 0.0,
        "other_non_res": 0.0224852071005917,
        "wind": 0.474222222222222,
        "pv": 0.22,
        "other_res": 0.202365308804205,
        "batteries": 0.157142857142857,
        "ror": 0.220793476556374,
        "hydro_turbine": 0.00816655164481252,
        "demand": 0.162055583375063,
        "hs_inflow": 0.00816655164481252, #0.00672879164625871,
        "phs_inflow": 0.00816655164481252, #0.00123855958530617,
    },

    "OBEROESTERREICH": {
        "gas": 0.2585,
        "coal": 0.0,
        "other_non_res": 0.552662721893491,
        "wind": 0.033,
        "pv": 0.174583333333333,
        "other_res": 0.236530880420499,
        "batteries": 0.228571428571429,
        "ror": 0.31519523286812,
        "hydro_turbine": 0.0628019323671498,
        "demand": 0.146532298447672,
        "hs_inflow": 0.0628019323671498, #0.0109316984283834,
        "phs_inflow": 0.0628019323671498, #0.0,
    },

    "SALZBURG": {
        "gas": 0.0303,
        "coal": 0.0,
        "other_non_res": 0.021301775147929,
        "wind": 0.0,
        "pv": 0.02675,
        "other_res": 0.0486202365308804,
        "batteries": 0.0714285714285714,
        "ror": 0.0393602007213423,
        "hydro_turbine": 0.254083275822406,
        "demand": 0.0791812719078618,
        "hs_inflow": 0.254083275822406, #0.0223307151337377,
        "phs_inflow": 0.254083275822406, #0.153118272656636,
    },

    "STEIERMARK": {
        "gas": 0.2739,
        "coal": 0.0,
        "other_non_res": 0.346745562130177,
        "wind": 0.121555555555556,
        "pv": 0.166666666666667,
        "other_res": 0.0867279894875164,
        "batteries": 0.0857142857142857,
        "ror": 0.0967539595421044,
        "hydro_turbine": 0.0730388773867035,
        "demand": 0.155232849273911,
        "hs_inflow": 0.0730388773867035, #0.0427862417383642,
        "phs_inflow": 0.0730388773867035, #0.0,
    },

    "TIROL": {
        "gas": 0.0190,
        "coal": 0.0,
        "other_non_res": 0.0,
        "wind": 0.000222222222222222,
        "pv": 0.0635,
        "other_res": 0.102496714848883,
        "batteries": 0.128571428571429,
        "ror": 0.135330092519994,
        "hydro_turbine": 0.212215320910973,
        "demand": 0.0925763645468202,
        "hs_inflow": 0.212215320910973, #0.761794860301214,
        "phs_inflow": 0.212215320910973, #0.10363260280179,
    },

    "VORARLBERG": {
        "gas": 0.0,
        "coal": 0.0,
        "other_non_res": 0.0,
        "wind": 0.0,
        "pv": 0.0203333333333333,
        "other_res": 0.0131406044678055,
        "batteries": 0.0285714285714286,
        "ror": 0.0128587109926298,
        "hydro_turbine": 0.0296756383712905,
        "demand": 0.0641587381071607,
        "hs_inflow": 0.0296756383712905, #0.143892621358456,
        "phs_inflow": 0.0296756383712905, #0.0873814995752665,
    },

    "WIEN": {
        "gas": 0.3909,
        "coal": 0.0,
        "other_non_res": 0.0,
        "wind": 0.00833333333333333,
        "pv": 0.071,
        "other_res": 0.0972404730617608,
        "batteries": 0.128571428571429,
        "ror": 0.0272855574721656,
        "hydro_turbine": 0.0,
        "demand": 0.204682023034552,
        "hs_inflow": 0.0,
        "phs_inflow": 0.0,
    },
}



