# 身体スペクトル適応度の最小参照結果

日付: 2026-09-26。状態: F1/F2の最小参照試験を実施。通常runtimeへの接続、F1/F2全体の完了、作者採用を意味しない。

## 版・入力・検証

基準sourceは `06a4772`、実装は隔離worktree `.worktrees/body-aware-fitness` にある。
取得前の条件は [参照契約](body-aware-fitness-reference.md)、作業全体は [実装計画](body-aware-fitness-plan.md) を参照。
sourceのhash、構造化した数値、全cargo testのログと終了状態は
[`target/body-fitness-reference-20260926-v5/`](../../target/body-fitness-reference-20260926-v5/) に保存した。
最初の取得は `target/body-fitness-reference-20260926/` に保持し、独立レビュー後の追加検査をv2、通常解析設定をv3、固定候補の応答をv4、時変ToneとPitchCore単一判断をv5として分けた。
`cargo test -- --nocapture` はexit 0。`cargo fmt --all -- --check`も通過した。
拡張した `cargo clippy --all-targets -- -D warnings` は、基準sourceと同じ5ファイルにある既存のtest lint 17件で失敗した。
この失敗は全対象lintの未通過として保持し、新規参照計算の合格へ含めない。
通常の `cargo clippy -- -D warnings` と `cargo check --all-targets` は通過した。

代表身体は実Toneを候補基音で再合成し、各frameを現行のNSGT・SpectralFrontEndへ通した後に密度を平均する。
自己除去はScheduleRendererが保存するsource単位のhabitat PCMを解析前に減算し、独立rendererの他者-only入力と比べた。
Sine/Harmonic/Modal、同音他者、別音他者、重なったTone、presentation-only、release tail、新source開始時の旧sourceのtailで、
登録したPCMおよび解析後の誤差条件を満たした。単独sourceの除去ではPCM、前処理密度、raw H/Rがゼロになる。

## 旧自己除去と評価方式の四条件

Harmonic、基音436.203064 Hz、同音の他者はSine。数値はscoreであり、sigmoid後のlevelではない。
環境の自己除去と候補身体の評価方式を交差させた。

| brightness | 同音他者 | 旧環境・点 | 旧環境・身体平均 | 他者-only・点 | 他者-only・身体平均 |
|---:|---|---:|---:|---:|---:|
| 0.2 | なし | 0.977253 | 0.926752 | 0.000000 | 0.000000 |
| 0.2 | あり | 0.977220 | 0.926716 | 1.000000 | 0.947691 |
| 0.9 | なし | 1.000000 | 0.835313 | 0.000000 | 0.000000 |
| 0.9 | あり | 0.977979 | 0.765519 | 1.000000 | 0.727767 |

単独sourceでも旧`ExactScan`は正の地形を残す。同音他者を含む場面も、他者だけを合成した参照と一致しない。
したがって「基音ビンの消去がsourceの除去に相当する」という仮定は、この固定条件で成立しない。
明るい身体ほど行動上の自己アンカーが強いという単調な主張は、この表からは出せない。
受理率・移動距離の実験では、候補集合・温度・移動費用・放射RMS・同じ乱数状態を別途固定する必要がある。

独立レビュー後、上の4場面それぞれでもmix minus own PCMから得た解析結果を他者-only参照へ直接照合し、登録許容を満たした。
さらに同音Sine二個体の反例では、前処理後の `max(0, q_mix-q_self)` と `q_other` の正規化L1差は0.770910となった。
したがって、PCM段階の自己除去を前処理済み密度の減算へ置き換えることは、この条件で成立しない。

## 身体平均の意味と帯域端

単一ビンと点評価の一致、不均一なERB幅での等質量平均、上部部分音だけが異なる身体対、
正の質量scaleに対する不変性、score平均後にsigmoidを適用する順序を検査した。
raw Rの身体積分は、同じ向き・環境正規化を使う部分音対二重和と一致し、LUTを読まないf64連続式の参照とも登録許容内で一致した。
raw Rを平均してから飽和・HR合成する案は、既存Cを身体平均する案と異なる値になった。
どちらもそのまま対称な実振幅積のSethares不協和度とは呼べない。

別の12場面では、Harmonicと非調和Modalを狭い分析帯域と広い分析帯域で比較した。
人工Cを「3500 Hz以下で0、それより上で-1」とすると、基音1800 Hzの身体平均は狭い帯域で0、
広い帯域で約-0.50〜-0.67になった。分析帯域外の質量を落として再正規化すると、
そこにある不利な寄与を評価しないという限界を具体化する反例である。
これは実音H/Rの地形や音域移動の観測ではない。異なる解析帯域の密度質量比も放射powerの保存率ではない。

## 通常の解析設定での確認

v3ではruntimeの既定解析器構築経路をそのまま使い、48 kHz、FFT=16384、hop=512、
55–8000 Hz・96 bins/oct・右寄せkernel・平滑化10 msで9場面を検査した。
Sine/Harmonic/Modalそれぞれで他者なし／440 Hz他者／466 Hz他者を64 hop観測し、
混合rendererのPCMから独立した自己-only rendererのPCMを引く結果を、他者-only rendererへ照合した。
密度の正規化L1差の最大は `1.390207e-7`、raw H/RおよびCの最大絶対差は `9.536744e-7` で、
事前の1e-3を下回った。単独除去は厳密なゼロであり、他者の寄与は残った。

この追加試験は通常設定の数値照合である。renderer内のsource採取境界やroute/tailは先行F1の別検査であり、
このv3で実時間workerへの配送や処理予算まで検証したわけではない。

## 固定候補に対する移動応答

v4では、7個の固定候補（現位置と±24/60/120セント）を使い、6移動候補を同確率で提案する
Metropolis則の期待受理率と期待距離を計算した。5段階のbrightness、通常gain／放射RMS=0.1、
他者なし／1.5倍基音のSine他者ありを交差させ、20場面の旧ExactScan／他者-only × 点／身体平均の
80レコードを得た。温度0.05/0.2を保存し、保存済み候補scoreからPythonでも期待値160組を照合した。
ここでは既存PitchCoreの候補抽出・greedy閾値・乱数消費は実行していない。

以下は温度0.05の期待距離（セント）。各セルはbrightness=0.1から0.9への端点であり、中間点も生データに残した。

| 外部音 | gain条件 | 旧環境・点 | 旧環境・身体平均 | 他者-only・点 | 他者-only・身体平均 |
|---|---|---:|---:|---:|---:|
| なし | 通常 | 0.00105 → 0.000872 | 3.907 → 1.903 | 68 → 68 | 68 → 68 |
| なし | RMS固定 | 0.00105 → 0.000872 | 3.907 → 1.903 | 68 → 68 | 68 → 68 |
| 1.5倍基音 | 通常 | 0.306 → 0.005 | 4.342 → 2.331 | 0.311 → 0.311 | 4.351 → 3.319 |
| 1.5倍基音 | RMS固定 | 0.311 → 0.311 | 4.351 → 3.319 | 0.311 → 0.311 | 4.351 → 3.319 |

単独sourceでは、正しい自己除去後のCはゼロで、全候補の期待受理率が1、期待距離が68セントになる。
旧環境の点評価では期待受理率が約3.63〜4.35×10⁻⁵へ下がり、音量固定でも残る。
したがって、この固定提案分布には自己アンカーの作用がある。brightnessに対してほぼ平坦な点評価でも、
自己アンカー自体が小さいとは言えない。元の反証条件の「平坦ならExactScan批判を撤回」は適切でない。

一方、外部音あり・RMS固定では旧環境と他者-onlyの差がこの精度ではほぼ消えた。
自己音の寄与は相対音量とピーク選択にも依存する。身体平均のbrightness依存も中間点で非単調であり、
明るい身体ほど常に移動しにくい、という一般則へ広げない。
この固定候補試験だけから実Voiceの提案受理率や長期の選択は主張しない。

## 時変Toneと既存PitchCoreへの追加対照

v5の3場面では、Sine/Harmonic/Modalそれぞれでmid-hopのUpdate、重なった新Toneのroute変更、releaseを検査した。
実際のcurrent frequencyは440 Hzから最初のglide frameで574.038 Hz、4 frame後に659.981 Hzへ動いた。
旧Toneのhabitat routeとrelease tailは残り、同sourceのpresentation-only Toneをhabitatから減算しなかった。
source採取PCMは独立した自己-only合成と一致し、除去後の解析は他者-only解析と登録許容内で一致した。
これはrouteが各Toneに属する既存APIの検査であり、body generationのworker配送を検証したものではない。

同じ20固定場面で、局所窓内のscore上位bin抽出・Gaussian候補・greedy閾値・Metropolis則を含む実際の
`PitchHillClimbPitchCore::propose_with_scorer`を、各セル64 seed×2温度で呼んだ。
身体評価は問い合わせられた各基音で代表Toneを再合成し、補間や格子丸めを使わない。
全80セル、計10,240判断を保存し、既存v4の数値はすべて一致した。

温度0.05・単独sourceでは、旧環境の点評価の移動は各brightnessで1/64回、平均距離0.057セントだった。
他者-onlyでは点／身体平均とも64/64回、平均90.915セントだった。
旧環境の身体平均では9〜11/64回、平均1.881〜2.631セントとなった。
通常gainとRMS固定の結果はこの単独条件で同じである。
旧点評価のbrightness依存が64 seedで完全に平坦でも、正しい自己除去との差が大きく残る。

外部音あり・RMS固定では両環境の選択結果が一致した。通常gainでは差があり、全条件を単調なbrightness効果で説明できない。
この比較は同じ初期位置からの単一判断を反復したもの。実Voiceの連続する運動、代謝・出生、
非同期計算期限の影響や長期の選択を含まない。F3の実時間配送の完了とは扱わない。

独立レビューでは、`top_local_peaks` の実体が窓内のscore上位10 binの抽出である点も確認した。
自己除去後の無音環境では全binが同点なので、この候補構成と同点時のMetropolis則が64/64の移動へ寄与する。
平均90.915セントを自然な無アンカー運動の基準値とせず、登録した候補生成器における一点判断の値として扱う。
固定候補160組の期待値と、実PitchCoreの10,240判断は独立検算済み。

## 独立レビュー

別担当がToneの代表条件、source・route・tailと解析履歴、密度平均の順序、ERB質量、sigmoidの順序、
raw R二重和と帯域対照をsource・保存結果から確認した。重大な数値不整合の指摘はなく、
当初未検査だった4場面のPCM直接比較と非線形密度減算の反例を上記v2で補った。
レビュー担当は保存済みのtest終了状態を確認したが、独立した機器や実装で全実験を再取得したわけではない。

## 残る工程

- 実時間のsource別解析状態、支持・利用可能時刻・世代・routeの配送と費用は未検証。
- 通常解析設定の固定場面、軽量設定でのpitch glideと後続Toneのroute変更まで検査済み。動的身体の代表密度とgeneration配送は未検証。
- 非調和身体が実際のH/R環境で作る候補順位・音程分布、帯域端の支持基準は継続検査する。
- 移動への接続はF3、代謝・出生・親選択への接続はF4。現時点で音色選択や遺伝の実装完了とはしない。
- 公開technoteの既存機構の説明だけを訂正した。ここに示す実験版を公開版の実装済み機構として加えていない。

移動へ接続する前の候補生成・解析履歴・配送時計の確認は [runtime引継ぎ](body-aware-fitness-runtime-handoff.md) に分けた。

## F3a: 最大4 sourceの解析状態

[事前登録](body-aware-fitness-f3a-registration.md)に従い、共有解析一つと最大4 sourceの自己除去解析を
所有する処理器を隔離版の `cfg(test)` 範囲へ実装した。出生frameの処理前に共有履歴を複製し、
中途出生、退役音のtail、slot再利用を独立した他者-only解析へ照合した。
容量超過・遅い登録・重複・epoch違い・frame欠落／逆行・不正PCMを理由別に拒否し、
拒否後に同じ観測履歴が自然復活しないことも検査した。

通常のruntime構築関数と実Toneによる4 source × 64 hopの比較では、密度の正規化L1誤差は
最大1.2263×10⁻⁷、H/R/Cの最大絶対誤差は9.5368×10⁻⁷だった。全数値許容内。
core層の合成PCMによる状態遷移検査と、この実Toneの参照検査は別に保存した。
2026-09-26 21:08:37 JSTの全 `cargo test -- --nocapture` はexit 0。
source、登録、全テスト、4数値記録は `target/body-fitness-f3a-20260926/` に保存した。

これは有界な解析処理器の検査であり、通常workerへの配線、候補身体の配送、
実時間の鮮度・費用、移動・生存への採用はまだ含まない。

## F3設計用の費用予備測定

[取得前登録](body-aware-fitness-cost-probe-registration.md)に基づくrelease計測は2 test、14条件ともexit 0。
`body-aware-fitness/target/f3-body-fitness-cost-20260926T212151JST/` にsource・binary・入力・ログを保存した。
代表身体密度は3身体×4基音、各72 hopの一呼出しが中央値42.0〜43.4 msだった。
これは候補一つの一括計算時間であり、1 hopの処理時間ではない。

自己除去Processorは1 sourceで中央値0.969 ms、p95 0.997 ms、4 sourceで中央値2.966 ms、
p95 3.288 ms、最大3.711 msだった。4 sourceの値は10.667 msのhop時間に対し
それぞれ27.8%、30.8%、34.8%に当たる。queue、候補評価、通常runtimeの総費用は含まない。

取得後の点検で、4 source中の最小自己PCM RMSが6.30×10⁻⁹であると判明した。
holdを10秒へ延ばしても `continuous_drive=0` の身体は減衰するため、4 source全ての持続励振を
測ったとは呼ばない。登録したRMS>0は満たすが、それだけでは十分な有音負荷の条件にならなかった。
混合・LOO解析は有音で、4 sourceのLOO mass最小は4.459、単独source除去のmassは0だった。
この限定を保持し、駆動を明示した追加取得で現結果を上書きしない。

この費用は候補ごとの同期解析を通常の音声処理へ置かない方針を支持する。
まず、密度確定後の不要なR/H計算を省く[最適化の数値保存検査](body-aware-fitness-density-only-registration.md)と、
有界workerの配送を別々に実装する。速くなることも、通常runtimeで期限に間に合うことも未判定である。

## F3b〜d: 配送・実Voice観測・一判断への接続

取得前登録に従い、有界workerの実thread試験、密度専用経路の同一性試験、実Voiceの観測試験、
候補準備と一判断への接続試験を通過した。密度専用経路はSine/Harmonic/Modal各4基音の毎frame・平均、
reset、tail、前処理パラメータ変更について旧full解析とbit一致した。費用改善量は別の専有取得で測る。

F3cはseed 1/4と1/4 sourceの4条件で48 hopずつ受理した。`wire_runtime` が作った実Voiceと
ScheduleRendererから自己PCMを採取し、独立した他者-only解析との密度L1誤差は最大1.21578×10⁻⁷、
場の最大絶対誤差は4.76837×10⁻⁷だった。観測ON/OFFのhabitat・presentation両bus音声はbit一致した。
実threadの経路を通しているが、配送は明示offlineで待ち合わせており、実時間の100 ms鮮度達成とはしない。

F3dはHarmonic 440 Hzの実VoiceとSine 466 Hzの外部音を固定し、環境を読む前に代表身体72 hopから
68候補を準備した。64 hopの自己除去環境からF2のscore表を作り、実PitchControllerのgateを通して消費した。
身体scoreは81回使われ、目標・commit後の周波数は466.163635 Hzになった。旧Cだけを大きく変えた対照でも
目標・salience・乱数状態は同じだった。gateが閉じた段階では表と乱数を消費しなかった。
候補不足とRNG・target・座標の不一致の4反例は、実RNGの追加消費前に拒否した。

全 `cargo test -- --nocapture` は22:44:09 JSTにexit 0、formatと標準Clippyもexit 0。
all-targets Clippyは既存テスト箇所17件でexit 101、新規F3b・費用probeへの指摘は解消した。
変更19 source、登録、全ログとSHAは `.worktrees/target-body-fitness/f3d-source-capsule-20260926/` に保存した。
この一判断は固定身体・固定制御・予約中の状態不変を前提とする。身体や制御の更新を含む非同期失効、
連続運動、有効評価率、代謝・出生への接続は残る。

## F3追加費用: 密度専用経路と明示drive付きProcessor

[密度専用経路の登録](body-aware-fitness-density-only-registration.md)と
[明示driveの登録](body-aware-fitness-driven-cost-registration.md)を、F3d検査後に固定した同一release lib test binaryで取得した。
封印binaryのSHA-256は `4767009cf6ace0d0f0ab6f83d26adb1fa6abc3ef00e0bcc5621d8d12bfbb79d0`。
223 sourceファイルのhash、source tar、登録2文書、build・実行command、全log、終了コード、結果と
独立再計算の補足JSONは `target/body-fitness-f3-extra-cost-prep-20260926-2244/` に保存した。
取得時のsource一致は同取得先の`recovery.json`で確認済み。以降の隔離版変更をこの測定のsourceへ遡及させない。

代表身体密度はSine／Harmonic／Modal×220／440／880／1760 Hzの12条件。
各条件で旧full解析と密度専用解析を一回ずつwarmupし、AB5対・BA5対、各群10標本を計時した。
同一入力の各frameと平均密度・massはbit一致し、各条件の10組でもfootprintはbit一致した。
一候補を72 hopで作る呼出しの中央値は、旧full解析41.567～42.785 ms、密度専用12.080～12.722 ms。
対差の中央値は12条件とも旧full側が29.472～30.083 ms長い。
この12 ms台は**一候補・72 hop全体**の計算時間であり、音声処理の一hop費用ではない。

明示driveの追加probeは1 source／4 sourceについて各3反復、64 hop warmup後の256 hop、
各768定常標本と出生3標本を取得した。計時区間の各own PCM最小RMSは1 sourceで0.2428、
4 sourceでは0.2428／0.6269／1.4589／0.2461で、登録下限1e-4をすべて超えた。
混合PCMも有音、単独sourceの自己除去massは0、4 sourceの最小massは12.546。

| source数 | 定常中央値 | p95 | 最大 | 出生中央値 |
|---:|---:|---:|---:|---:|
| 1 | 0.954 ms | 0.964 ms | 1.018 ms | 0.762 ms |
| 4 | 2.960 ms | 3.388 ms | 3.455 ms | 2.964 ms |

10.667 msのhop予算に対し、4 source定常の中央値／p95／最大は27.8%／31.8%／32.4%。
両probeのtestはexit 0。最初の計時JSONだけlibtestのtest名接頭辞と同じ行に出たため、
元の`capture.py`は密度12件中11件しか抽出せず停止した。元logと11件の部分結果を保持し、
接頭辞を厳密に除いた復旧で元logから12件を回収した。密度の数値再取得はせず、未実行だったdrivenだけを
同じ封印binaryで実行した。`recovery.json`は両probeの完了、`independent-verification.json`は
生logと復旧結果、封印source・binary・登録のhash、標本数・統計の独立照合を記録する。

旧14条件の費用probeは別時刻・別版の資料として維持する。今回の明示driveにより先行4 source条件の
持続励振不足は補ったが、queue、通常runtime全hop、音声callback、候補評価、有効期限、16／64 sourceの
資源受入は含まない。F3全体の実時間合格や通常採用を宣言しない。
