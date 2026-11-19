export const data = {
  "status": "Illustrative replay-path calculation; not new inference measurements.",
  "date": "2026-11-18",
  "step_ms": 18.9,
  "p": 0.98,
  "lookup_ms": 0.378,
  "winner": 8,
  "rows": [
    {
      "k": 0,
      "input_tokens": 1,
      "v_ms": 18.9,
      "l_ms": 0.0,
      "r_ms": 0.0,
      "g_tokens": 1.0,
      "c_ms_per_token": 18.9,
      "ratio": 1.0,
      "v_display": "18.90",
      "ratio_display": "1.00x"
    },
    {
      "k": 2,
      "input_tokens": 3,
      "v_ms": 20.79,
      "l_ms": 0.378,
      "r_ms": 1.118880000000001,
      "g_tokens": 2.9404,
      "c_ms_per_token": 7.579540198612434,
      "ratio": 1.1,
      "v_display": "20.79",
      "ratio_display": "1.10x"
    },
    {
      "k": 4,
      "input_tokens": 5,
      "v_ms": 23.058,
      "l_ms": 0.378,
      "r_ms": 3.6310559040000028,
      "g_tokens": 4.80396016,
      "c_ms_per_token": 5.634321476970784,
      "ratio": 1.22,
      "v_display": "23.06",
      "ratio_display": "1.22x"
    },
    {
      "k": 8,
      "input_tokens": 9,
      "v_ms": 27.972,
      "l_ms": 0.378,
      "r_ms": 12.393574645846634,
      "g_tokens": 8.312611893492505,
      "c_ms_per_token": 4.901416686822896,
      "ratio": 1.48,
      "v_display": "27.97",
      "ratio_display": "1.48x"
    },
    {
      "k": 16,
      "input_tokens": 17,
      "v_ms": 38.745,
      "l_ms": 0.378,
      "r_ms": 42.13472333292072,
      "g_tokens": 14.533911690967702,
      "c_ms_per_token": 5.590905260792209,
      "ratio": 2.05,
      "v_display": "38.75",
      "ratio_display": "2.05x"
    }
  ],
  "r_formula": "T_step * sum((j+1) * p**j * (1-p), j=0..k-1)",
  "g_formula": "1 + sum(p**j, j=1..k)",
  "c_formula": "(V(k) + L + R(k)) / G(k)",
  "replay_example": {
    "accepted": 2,
    "ordinary_forwards": 3,
    "conditional_recovery_ms": 56.7
  },
  "source": {
    "baseline": "User-selected historical reference scale; the archived study is withdrawn.",
    "archived_path": "benches/agent_study_20261021_main/calibration.json",
    "archived_sha256": "b85d8a589ddccf5014895dcc26da59f9dbb67fa6c9ff4f21d9610c543448aaa4",
    "archived_step_ms": 18.900229479186237,
    "archived_verification_ms_not_used": {
      "2": 19.53047898132354,
      "4": 24.604666512459517,
      "8": 36.78172902436927,
      "16": 43.0810414836742
    },
    "ratios_path": "projects/siliconllm_2026_11_11_predefense/sources/illustrative_placeholders.json",
    "ratios_sha256": "f876a674e487ecb8f5cefc5b82c4389b9db2278504915c2fa4602c21e4cddec0",
    "ratios_are_measurements": false
  },
  "assumptions": [
    "A constant, independent per-token match probability p is an approximation.",
    "Rejected batches replay j+1 ordinary forwards after j accepted candidates.",
    "Full acceptance has no extra replay in this example.",
    "EOS, output-budget truncation and additional policy overhead are not modeled here.",
    "Native trim requires its own measured recovery-cost model."
  ]
};
