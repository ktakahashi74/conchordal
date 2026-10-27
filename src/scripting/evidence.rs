//! Evidence metadata only: no execution defaults or composer controls.

/// Each entry describes one part of a term independently of its disclosure tier.
pub struct EvidenceDoc {
    pub en: &'static str,
    pub ja: &'static str,
}

pub struct ConfigDoc {
    pub name: &'static str,
    pub evidence: &'static [EvidenceDoc],
}

pub const BODY: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Not a mechanism (initial body): synthesis traits and initial conditions; see the selected body and behavior terms (§9.3.56).",
    ja: "該当なし（初期の身体）：合成特性と初期条件。選んだ身体と行為の語を参照（§9.3.56）。",
}];

pub const PLACEMENT: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Not a mechanism (placement): initial position, extent or population count (§9.3.56).",
    ja: "該当なし（配置）：開始位置、範囲、個体数（§9.3.56）。",
}];

pub const FIELD_PLACEMENT: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Not a mechanism (placement): reads H/R and the authored Consonance Field valuation; see set_roughness_aversion and set_pitch_objective.",
        ja: "該当なし（配置）：H/Rと作者が定めたConsonance Fieldの評価を入力とする。set_roughness_aversion、set_pitch_objectiveを参照。",
    },
    EvidenceDoc {
        en: "Principle: auditory H/R functions as input; this does not certify every kernel equation or constant (§9.3.56, §9.3.104).",
        ja: "原理：入力となるH/Rの感覚機能。個々の式・定数の原典一致を認定する札ではない（§9.3.56、§9.3.104）。",
    },
    EvidenceDoc {
        en: "Author rule: extrema, density, spacing or tension-based selection (§9.3.56).",
        ja: "作者規則：極値・密度・間隔・緊張に基づく選択（§9.3.56）。",
    },
];

pub const FORM: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Not a mechanism (macro form): scenario timing, placement or release; input mechanisms retain their own labels.",
    ja: "該当なし（大きな形）：Scenarioの時間構成、配置、解放。入力の機構の札は各語を参照。",
}];

pub const ROUTING: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Not a mechanism (routing): selects or combines buses; see leave_self_out for the independent self-perception rule.",
    ja: "該当なし（経路）：busの選択・結合。独立した自己知覚の規則はleave_self_outを参照。",
}];

pub const OPERATION: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Not a mechanism (operation): output, reproducibility or research execution settings (§9.3.56).",
    ja: "該当なし（運転）：出力、再現性、研究用の実行設定（§9.3.56）。",
}];

pub const CONTAINER: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Not a mechanism (API container): use the labels of the constructor and operations on this type.",
    ja: "該当なし（APIの器）：構築・操作に使う語の札を参照。",
}];

pub const PHONATION: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Author rule: phonation, participation, duration or articulation policy (§9.3.57); existing base defaults remain active.",
    ja: "作者規則：発声、参加、持続、再発音の方策（§9.3.57）。既存の基本動作の既定値は維持。",
}];

pub const PITCH: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Author rule: pitch movement, proposal selection or search cost (§9.3.57, §9.3.98–99); existing base defaults remain active.",
    ja: "作者規則：音高移動、提案選択、探索費用（§9.3.57、§9.3.98–99）。既存の基本動作の既定値は維持。",
}];

pub const SEEK: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Principle: auditory H/R functions as input; this does not certify every kernel equation or constant (§9.3.56, §9.3.104).",
        ja: "原理：入力となるH/Rの感覚機能。個々の式・定数の原典一致を認定する札ではない（§9.3.56、§9.3.104）。",
    },
    EvidenceDoc {
        en: "Author rule: consonance valuation and local search, including weights and glide defaults (§9.3.98).",
        ja: "作者規則：協和の評価と局所探索。重みとglideの既定値を含む（§9.3.98）。",
    },
];

pub const SELF_EXCLUSION: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Principle: auditory H/R functions as input; this does not certify every kernel equation or constant (§9.3.56, §9.3.104).",
        ja: "原理：入力となるH/Rの感覚機能。個々の式・定数の原典一致を認定する札ではない（§9.3.56、§9.3.104）。",
    },
    EvidenceDoc {
        en: "Author rule: whether own habitat PCM is excluded from H/R; Voice defaults false, Flora true. Flora resource-power exclusion is independent (§9.3.95).",
        ja: "作者規則：H/R入力から自己のhabitat PCMを除く選択。Voiceの既定はfalse、Floraはtrue。Floraの資源powerの自己除外とは独立（§9.3.95）。",
    },
];

pub const METER: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Principle: beat perception as input (§9.3.56).",
        ja: "原理：入力となる拍の知覚機能（§9.3.56）。",
    },
    EvidenceDoc {
        en: "Candidate (source match pending): Large neural-resonance oscillator form and Large & Jones (1999) simulation gains; source agreement and candidate acceptance are not established here.",
        ja: "候補（照合未了）：Largeの神経共鳴振動子の形とLarge & Jones（1999）のシミュレーション利得。原典一致・候補受入を認定しない。",
    },
    EvidenceDoc {
        en: "Author rule: participation, coupling defaults and priors; metric uses entrainment 0.95, meter_stability defaults 0 and an unspecified basin centers on 2 Hz. These values are not derived principles.",
        ja: "作者規則：参加、結合の既定値、事前分布。metricのentrainmentは0.95、meter_stabilityの既定は0、basin未指定時の中心は2 Hz。これらの値は原理から導いた定数ではない。",
    },
];

pub const PARTICIPATION: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Principle: beat perception as input (§9.3.56).",
        ja: "原理：入力となる拍の知覚機能（§9.3.56）。",
    },
    EvidenceDoc {
        en: "Author rule: participation, skip and recurrence policies and coupling values; intrinsic_rate supplies an initial body scale (§9.3.55–57).",
        ja: "作者規則：参加・skip・再帰入力の扱いと結合値。intrinsic_rateは身体の初期尺度（§9.3.55–57）。",
    },
];

pub const HABITUATION: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Principle: habituation as an input function (§9.3.60).",
        ja: "原理：入力となる馴化の機能（§9.3.60）。",
    },
    EvidenceDoc {
        en: "Author rule: current habituation erosion law and engineering time/drive constants (§9.3.60).",
        ja: "作者規則：現行の馴化の浸食式と時間・駆動の工学値（§9.3.60）。",
    },
];

pub const REST_HABITUATION: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Principle: habituation as an input function (§9.3.60).",
        ja: "原理：入力となる馴化の機能（§9.3.60）。",
    },
    EvidenceDoc {
        en: "Author rule: current habituation erosion law and engineering time/drive constants (§9.3.60).",
        ja: "作者規則：現行の馴化の浸食式と時間・駆動の工学値（§9.3.60）。",
    },
    EvidenceDoc {
        en: "Author rule: probabilistic span exit and private recovery/return; neutral at weight 0 (§9.3.96, §9.3.101).",
        ja: "作者規則：spanの確率的な退出と私的回復・復帰。重み0で中立（§9.3.96、§9.3.101）。",
    },
];

pub const DWELL: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Principle: auditory H/R functions as input; this does not certify every kernel equation or constant (§9.3.56, §9.3.104).",
        ja: "原理：入力となるH/Rの感覚機能。個々の式・定数の原典一致を認定する札ではない（§9.3.56、§9.3.104）。",
    },
    EvidenceDoc {
        en: "Principle: habituation as an input function (§9.3.60).",
        ja: "原理：入力となる馴化の機能（§9.3.60）。",
    },
    EvidenceDoc {
        en: "Author rule: current habituation erosion law and engineering time/drive constants (§9.3.60).",
        ja: "作者規則：現行の馴化の浸食式と時間・駆動の工学値（§9.3.60）。",
    },
    EvidenceDoc {
        en: "Author rule: proposal delay driven by the habituated field level; neutral at weight 0 (§9.3.98).",
        ja: "作者規則：馴化後の場の水準に応じた提案の遅延。重み0で中立（§9.3.98）。",
    },
];

pub const ECOLOGY: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Author rule: metabolism, viability, neighbor interaction or lifecycle/respawn; sensory input does not establish an auditory model for this ecology (§9.3.57).",
    ja: "作者規則：代謝、生存可能性、近傍個体との相互作用、生存過程・再生成。感覚入力を使っても、生態則が聴覚モデルになるわけではない（§9.3.57）。",
}];

pub const FLORA_BODY: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Not a mechanism (initial body): Flora body, amplitude or fade choices; see flora and pace.",
        ja: "該当なし（初期の身体）：Floraの身体、振幅、fadeの選択。flora、paceを参照。",
    },
    EvidenceDoc {
        en: "Author rule: underlying spectral growth, consumption and retirement ecology; not an auditory cognition model.",
        ja: "作者規則：背後のスペクトル成長、消費、退出の生態則。聴覚認知のモデルではない。",
    },
];

pub const FLORA_PACE: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Principle: beat perception as input (§9.3.56).",
        ja: "原理：入力となる拍の知覚機能（§9.3.56）。",
    },
    EvidenceDoc {
        en: "Author rule: ecology speed, action opportunities, the default multiplier 1 and the 10-second ecological saturation per opportunity (§9.3.97–99, §9.3.102).",
        ja: "作者規則：生態時間の速度、行動機会、既定倍率1、機会あたり10秒の生態時間の飽和（§9.3.97–99, §9.3.102）。",
    },
];

pub const RAIN: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Author rule: impact texture; impacts drawn one by one from distributions and resonant objects they strike; a composition rule, not an auditory model (docs/design-notes/rain.md).",
    ja: "作者規則：分布から一つずつ引く打音と、打音が当たって鳴る共鳴する物によるtexture。聴覚モデルではない作曲上の規則（docs/design-notes/rain.md）。",
}];

pub const RAIN_BODY: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Not a mechanism (initial body): Rain amplitude, fade, impact material or object body; see rain.",
    ja: "該当なし（初期の身体）：Rainの振幅、fade、打音の素材、物の身体。rainを参照。",
}];

pub const RAIN_OBJECTS: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Principle: auditory H/R functions as input; this does not certify every kernel equation or constant (§9.3.56, §9.3.104).",
        ja: "原理：入力となるH/Rの感覚機能。個々の式・定数の原典一致を認定する札ではない（§9.3.56、§9.3.104）。",
    },
    EvidenceDoc {
        en: "Author rule: objects start on consonance peaks and drift toward consonance with the Voice movement-cost defaults when struck; anchor() holds them.",
        ja: "作者規則：物は協和の山から始まり、打たれるたびにVoiceの移動費用の既定値で協和の方へ動く。anchor()で固定する。",
    },
];

pub const SWAY: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Author rule: 1/f fluctuation of activity from nine AR(1) processes between 0.4 and 102 s; motivated by auditory-cortex tuning to 1/f dynamics (Garcia-Lazaro et al. 2006), not a model of it.",
    ja: "作者規則：0.4〜102秒の九つのAR(1)過程による活動の1/fの揺らぎ。聴覚野の1/f変動への同調（Garcia-Lazaro et al. 2006）を動機とするが、そのモデルではない。",
}];

pub const FIELD_BODY: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Not a mechanism (initial body): synthesis traits and initial conditions; see the selected body and behavior terms (§9.3.56).",
        ja: "該当なし（初期の身体）：合成特性と初期条件。選んだ身体と行為の語を参照（§9.3.56）。",
    },
    EvidenceDoc {
        en: "Principle: auditory H/R functions as input; this does not certify every kernel equation or constant (§9.3.56, §9.3.104).",
        ja: "原理：入力となるH/Rの感覚機能。個々の式・定数の原典一致を認定する札ではない（§9.3.56、§9.3.104）。",
    },
    EvidenceDoc {
        en: "Author rule: selecting body modes from landscape density or peaks; valuation remains authored.",
        ja: "作者規則：Landscapeの密度・極値から身体のモードを選ぶ規則。入力の評価も作者の規則。",
    },
];

pub const VALUATION: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Author rule: consonance evaluation, sign or roughness preference; does not alter H/R sensation (§9.3.56). Existing valuation defaults remain active.",
    ja: "作者規則：協和の評価、符号、粗さの好み。H/Rの感覚は変更しない（§9.3.56）。既存の評価の既定値は維持。",
}];

pub const COUPLING: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Author rule: coupling weight; Voice social coupling and listener DCC feedback are distinct paths (§9.3.57).",
    ja: "作者規則：結合の重み。Voice間の結合と聴き手のDCC帰還は別経路（§9.3.57）。",
}];

pub const RESEARCH: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Author rule: research model, comparison, normalization or control; not a composer control over internal learning (§9.3.55–57).",
    ja: "作者規則：研究用のモデル、比較、正規化、対照。内部学習を作曲者が操作する語ではない（§9.3.55–57）。",
}];

pub const SENSORY_SCALE: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Principle: auditory H/R functions as input; this does not certify every kernel equation or constant (§9.3.56, §9.3.104).",
        ja: "原理：入力となるH/Rの感覚機能。個々の式・定数の原典一致を認定する札ではない（§9.3.56、§9.3.104）。",
    },
    EvidenceDoc {
        en: "Author rule: adopted sensory implementation scale; the label neither certifies every coefficient nor opens layer-1 sensation to composing (§9.3.94).",
        ja: "作者規則：採用した感覚実装の尺度。全係数の原典一致の認定でも、層1の感覚を作曲用に開放する札でもない（§9.3.94）。",
    },
];

pub const DATA: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Not a mechanism (observation data): frozen descriptors, identifiers or acquisition geometry; data are not cognitive constants.",
    ja: "該当なし（観測データ）：凍結した記述子、識別子、取得時の構成。データを認知の定数としない。",
}];

pub const SEQUENCE: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Candidate (PPM-Decay, Harrison et al. 2020): fixed published-analysis model; source fixtures cover distributions, information content and learning state, not R API or RNG equivalence (§9.3.105).",
        ja: "候補（PPM-Decay、Harrisonほか2020）：公表解析の固定モデル。原典fixtureの照合範囲は分布・情報量・学習状態であり、R APIや乱数生成器の同値性ではない（§9.3.105）。",
    },
    EvidenceDoc {
        en: "Author rule: executed-interval encoding, execution adaptation and follow scoring after candidate generation; weight defaults 0. No default listener preset adoption (§9.3.105).",
        ja: "作者規則：実行された音程の符号化、実行への接続、候補生成後の追従評価。重みの既定値は0。既定の聴き手の型への採用は未実施（§9.3.105）。",
    },
];

pub const PHRASE_BOUNDARY: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Candidate (LBDM, Cambouropoulos 2001): local boundary model with pitch/IOI/rest weights .25/.50/.25; implementation checks do not reproduce the original program or corpus results (§9.3.106).",
        ja: "候補（LBDM、Cambouropoulos 2001）：音高・IOI・休みの重み.25/.50/.25を用いる局所境界モデル。実装の検査は原著プログラムやcorpus成績の再現ではない（§9.3.106）。",
    },
    EvidenceDoc {
        en: "Author rule: finite-sequence completion, six-onset online adaptation, birth-fixed caps and post-boundary gain 1/(1+w*B) on one later legal Sine onset. Weight defaults 0; no default listener preset adoption (§9.3.106).",
        ja: "作者規則：有限列の完備化、6 Onのオンライン接続、出生時に固定する上限、境界判明後の合法なSine On一件へのgain 1/(1+w*B)。重みの既定値は0。既定の聴き手の型への採用は未実施（§9.3.106）。",
    },
];

/// AppConfig leaf keys, including research-only optional records.
pub const CONFIG_DOCS: &[ConfigDoc] = &[
    ConfigDoc {
        name: "birth_surrogate",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "body_metabolism.enabled",
        evidence: ECOLOGY,
    },
    ConfigDoc {
        name: "body_metabolism.updates_per_hop",
        evidence: ECOLOGY,
    },
    ConfigDoc {
        name: "body_metabolism.observation_frames",
        evidence: ECOLOGY,
    },
    ConfigDoc {
        name: "body_metabolism.representative_hold_sec",
        evidence: ECOLOGY,
    },
    ConfigDoc {
        name: "render_prototype",
        evidence: OPERATION,
    },
    ConfigDoc {
        name: "temporal_meter_t2",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "audio.latency_ms",
        evidence: OPERATION,
    },
    ConfigDoc {
        name: "audio.sample_rate",
        evidence: OPERATION,
    },
    ConfigDoc {
        name: "audio.limiter",
        evidence: OPERATION,
    },
    ConfigDoc {
        name: "audio.output_gain_db",
        evidence: OPERATION,
    },
    ConfigDoc {
        name: "analysis.nfft",
        evidence: OPERATION,
    },
    ConfigDoc {
        name: "analysis.hop_size",
        evidence: OPERATION,
    },
    ConfigDoc {
        name: "analysis.tau_ms",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "analysis.kernel_align",
        evidence: OPERATION,
    },
    ConfigDoc {
        name: "psychoacoustics.loudness_exp",
        evidence: SENSORY_SCALE,
    },
    ConfigDoc {
        name: "psychoacoustics.roughness_k",
        evidence: SENSORY_SCALE,
    },
    ConfigDoc {
        name: "psychoacoustics.consonance.field.kernel.a",
        evidence: VALUATION,
    },
    ConfigDoc {
        name: "psychoacoustics.consonance.field.kernel.b",
        evidence: VALUATION,
    },
    ConfigDoc {
        name: "psychoacoustics.consonance.field.kernel.c",
        evidence: VALUATION,
    },
    ConfigDoc {
        name: "psychoacoustics.consonance.field.kernel.d",
        evidence: VALUATION,
    },
    ConfigDoc {
        name: "psychoacoustics.consonance.field.level.beta",
        evidence: VALUATION,
    },
    ConfigDoc {
        name: "psychoacoustics.consonance.field.level.theta",
        evidence: VALUATION,
    },
    ConfigDoc {
        name: "psychoacoustics.consonance.density.roughness_gain",
        evidence: VALUATION,
    },
    ConfigDoc {
        name: "psychoacoustics.habituation.enabled",
        evidence: HABITUATION,
    },
    ConfigDoc {
        name: "psychoacoustics.habituation.satiation_sec",
        evidence: HABITUATION,
    },
    ConfigDoc {
        name: "psychoacoustics.habituation.recovery_sec",
        evidence: HABITUATION,
    },
    ConfigDoc {
        name: "psychoacoustics.habituation.ref_drive",
        evidence: HABITUATION,
    },
    ConfigDoc {
        name: "psychoacoustics.use_incoherent_power",
        evidence: OPERATION,
    },
    ConfigDoc {
        name: "dcc.coupling_strength",
        evidence: COUPLING,
    },
    ConfigDoc {
        name: "dcc.max_temperature_bonus",
        evidence: COUPLING,
    },
    ConfigDoc {
        name: "playback.wait_user_exit",
        evidence: OPERATION,
    },
    ConfigDoc {
        name: "playback.wait_user_start",
        evidence: OPERATION,
    },
    ConfigDoc {
        name: "temporal_ridge.means",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_ridge.deviations",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_acoustic.group_means",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_acoustic.group_deviations",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_acoustic.accent_means",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_acoustic.accent_deviations",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_acoustic.group_retirement_sec",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_acoustic.inactive_energy_max",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_acoustic.correlation_window_sec",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_acoustic.min_pairs",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_acoustic.min_coverage",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_acoustic.persistence_hops",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_period.model",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_period.horizon_sec",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_t2.component_weights",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_t2.gain",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_t2.threshold",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_t2.weight_gain",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_t2.window",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_t2.combination",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_t2.attribution",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_t2.unknown",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_body.means",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_body.deviations",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_body.accent_means",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_body.accent_deviations",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_body_prototypes.model_version",
        evidence: DATA,
    },
    ConfigDoc {
        name: "temporal_body_prototypes.sample_rate",
        evidence: DATA,
    },
    ConfigDoc {
        name: "temporal_body_prototypes.nfft",
        evidence: DATA,
    },
    ConfigDoc {
        name: "temporal_body_prototypes.hop_size",
        evidence: DATA,
    },
    ConfigDoc {
        name: "temporal_body_prototypes.means",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_body_prototypes.deviations",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_body_prototypes.accent_means",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_body_prototypes.accent_deviations",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_body_prototypes.medoids.record_id",
        evidence: DATA,
    },
    ConfigDoc {
        name: "temporal_body_prototypes.medoids.raw_values",
        evidence: DATA,
    },
    ConfigDoc {
        name: "temporal_body_prototypes.medoids.mask",
        evidence: DATA,
    },
    ConfigDoc {
        name: "temporal_onset_comparison.footprint",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_onset_comparison.arrival",
        evidence: RESEARCH,
    },
    ConfigDoc {
        name: "temporal_onset_comparison.arrival_weight",
        evidence: RESEARCH,
    },
];

pub const EXPRESSION: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Candidate: Todd (1992) single-layer intensity and Friberg–Sundberg (1999) q=3 timing curve; not a verified brain mechanism.",
        ja: "候補：Todd (1992) の単層の強度と Friberg–Sundberg (1999) の q=3 の時間曲線。脳機構の確定ではない。",
    },
    EvidenceDoc {
        en: "Author rule: known final interval, weight saturation, amplitude floor, frozen generation clock and supported voice restrictions (§9.3.109).",
        ja: "作者規則：既知の最終区間、重みの飽和、振幅の床、生成時計の固定と対象の制限（§9.3.109）。",
    },
];

/// A profile selects mechanisms; it supplies no separate evidence claim.
pub const LISTENER_PROFILE: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Not a mechanism (profile selection): see follow_own_sequence, melodic_continuity, phrase_boundary and expressive_section for candidate models and author rules.",
    ja: "該当なし（型の選択）：候補モデルと作者規則の札は follow_own_sequence、melodic_continuity、phrase_boundary、expressive_section を参照。",
}];
pub const MELODIC_CONTINUITY: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Candidate: von Hippel (2000), text-consistent AR(1) mean and variance; not an author-approved erratum.",
        ja: "候補：von Hippel (2000) の本文に整合する AR(1) の平均・分散。著者公認の訂正ではない。",
    },
    EvidenceDoc {
        en: "Author rule: finite-prefix Pearson estimation, committed self-onset adapter, continuous log-density scoring and replacement of movement cost and fixed-center gravity (§9.3.108).",
        ja: "作者規則：有限 prefix の Pearson 推定、確定した自己 On の入力、連続音高の対数密度採点、移動費用と固定中心への重力の置換（§9.3.108）。",
    },
];

pub const TONAL_CONTEXT: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Candidate (source matched): IPEM sensory short-term memory, 40-channel ANI, periodicity pitch and 0.1/1.5-second half-lives. This implements a hypothesis, not an established brain mechanism.",
        ja: "候補（原典照合済み）：IPEMの感覚的短期記憶。40帯域ANI、periodicity pitch、半減時間0.1／1.5秒。仮説の実装であり、確定した脳機構ではない。",
    },
    EvidenceDoc {
        en: "Author rule: sine probe, tabulated approximation, raw correlation readout, candidate rescoring and dwell coupling; default weight 0 (§9.3.110).",
        ja: "作者規則：正弦probe、表による近似、相関の読み出し、候補再採点とdwellへの接続。重みの既定値は0（§9.3.110）。",
    },
];

pub const HARMONIC_RHYTHM: &[EvidenceDoc] = &[EvidenceDoc {
    en: "Author rule: shared opportunities for background changes. Zero is neutral; N is a composer quantity, not an inferred measure or a perceptual constant.",
    ja: "作者規則：背景の変化を共有の機会にまとめる。0は中立。Nは作曲者が置く量であり、推定小節長や知覚の定数ではない。",
}];

pub const PITCH_SET: &[EvidenceDoc] = &[
    EvidenceDoc {
        en: "Author rule: cumulative periodicity experience, continuous-lag adapter, peak proposals and scoring. Zero is neutral (§9.3.114).",
        ja: "作者規則：周期性の経験分布を累積し、その山を候補と採点へ戻す。連続化も作者規則であり、0は中立（§9.3.114）。",
    },
    EvidenceDoc {
        en: "Candidate (source): IPEM ANI/periodicity observation. The complete control is not a published learning model or an established brain mechanism.",
        ja: "候補（出典あり）：IPEMのANI・周期性観測。語全体が公表された学習モデルや確定した脳機構を実装するという意味ではない。",
    },
];
