# 身体スペクトル適応度 F1/F2 の最小参照契約

登録日: 2026-09-26。基準 source: `06a4772`。数値取得前の契約。
関連: [実装計画](body-aware-fitness-plan.md)、[現在地監査](body-aware-fitness-f0-audit.md)。

## 対象と実行場所

隔離 worktree `.worktrees/body-aware-fitness` で参照計算を作る。最初は crate 内の試験用経路に限り、
音声 callback、通常の pitch decision、代謝、respawn へ接続しない。これらを実装済みと呼ばない。
I11-1 の取得・判定中はビルド・数値試験を走らせない。取得後は専用の `CARGO_TARGET_DIR` を使う。

## 代表身体

実際の `Tone` を Recipe の body、hold、ADSR、modulator、sample rate から構築し、候補ごとの基音で再合成する。
代表振幅、kick、onset trigger は各1とし、現在の vitality・休符状態・親の包絡状態を持ち込まない。
位相 seed は source と固定した tone/onset 識別子から再現可能に作る。

観測時間は caller が指定する完全な hop の個数とする。各 hop の PCM を既存 `AnalysisStream` に通し、
各 frame の `subjective_intensity` 密度を処理後に時間平均する。平均 power の後から圧縮する方式とは区別する。
代表音の試験では解析状態を reset して同じ条件から始める。実際の自己除去にこの reset を転用しない。
最初の軽量試験設定は fs=8 kHz、FFT=1024、hop=256、Log2Space=100–3500 Hz、24 bins/oct、
coherent power、loudness exponent=0.23、ref power=1e-4、平滑化60 ms、24 frameとする。
これは数値契約の検査用であり、48 kHz の通常設定や実時間資源を検証したことにはならない。

出力の `in_band_mass` は密度の ERB 積分であり、全放射に対する帯域内支持率ではない。
Nyquist と分析帯域による損失をこの値だけから推定しない。F2 の帯域端比較は同じ身体・基音を
異なる分析帯域でも観測し、非線形前処理後の質量を音響 power の保存率と呼ばない。

## source 単位の自己除去

`ScheduleRenderer` の既存 source 採取境界で、全 Tone の route 後 habitat PCM を合計した値を使う。
同じ混合順序で他 source だけを合成した独立 renderer を比較対象とし、混合 PCM から自己 PCM を引いた入力を
NSGT 前に入れる。解析状態は同じ開始時刻から同じ hop 数を処理して揃える。

試験は Sine/Harmonic/Modal、単独個体、同音他者、別音他者、複数 Tone、habitat 無効の route、
release 後の tail、source の退役と新しい source の開始を含める。退役した source の tail は他者に残す。
別 source の同音成分を消す実装と、基音ビンだけ消す実装は合格させない。

PCM 差の事前許容は各 sample で `32 * f32::EPSILON * max(1, abs(mix), abs(own), abs(reference))`。
正解が厳密な無音の単独除去は、PCM と raw R/H と subjective density がゼロになることを別に要求する。
既存履歴に由来する habituation がゼロになることは要求しない。

解析後の差はまず同じ bins 上の密度の L1 誤差を参照の積分質量で正規化し、上限1e-3とする。
raw H/R と C の差は最大絶対誤差1e-3を最初の許容とする。
ピーク選択の不連続性によって不合格になった場合は、誤差と配置を保存して原因を説明し、
結果を見て許容を緩めた同じ試験を事前登録合格とは扱わない。必要な契約変更は別版・別取得にする。

## 身体評価

`w_b = q_b * du_b / sum(q_b * du_b)`、`fitness_score = sum(w_b * C_minus_i_eff_b)`、
`fitness_level = sigmoid(beta * (fitness_score - theta))` とする。
du は caller が Log2Space から得た ERB bin 幅を渡し、評価ループ内の allocation を避ける。
ゼロ質量・非有限値等の不適格入力は明示した unsupported とし、有効な score=0 と区別する。
全 `_scan` 境界は Log2Space 長を named hard assert で確認する。

検査対象は単一ビンと点評価の一致、不均一 du 上での等質量2点の平均、同じ基音で上部部分音だけが
衝突する身体対、正の質量 scale に対する不変性、score 集約後と level 集約後の違いである。
有限な小型演算の許容は絶対誤差1e-6、scale 不変性は1e-5とする。

raw R を積分してから飽和・HR 合成する別案も小型例で比較する。現行 C の身体平均と同じ量とは扱わない。
R の線形畳み込みと部分音の直接二重和は、同じ向きの kernel、環境正規化、ERB 質量を使って照合する。
独立した連続 kernel 式との照合では LUT 近似を含むため、相対誤差5e-4または絶対誤差1e-6を事前許容とする。
方向性 kernel と LUT 近似を含む現行モデルを、対称な実振幅積の Sethares モデルと同一視しない。

四条件の最初の実音比較は、Harmonic の brightness=0.2/0.9、それぞれ単独/同音の Sine 他者ありを使う。
分析格子の440 Hzに最も近い基音、24 frame、hold=8000 sample、個体振幅0.35、代表振幅1に固定する。
旧 `ExactScan` の実関数が返す環境と、他 source の独立合成による環境を、点評価/身体評価で交差させる。
旧関数が fallback に入る場合は比較の有効件数に数えない。最初の検査では登録した4場面すべてで
旧関数が有効に呼ばれることを要求する。結果は `BODY_FITNESS_STATIC` の JSON 行として試験ログに残す。
単独 source の除去がゼロになることと、旧方式に上部部分音由来の場が残ることを検査するが、
明るさに伴う行動上の移動量・受理率の単調な変化はこの静的比較から主張しない。

## F2 の追加対照（追加取得前、2026-09-26）

上の最小比較に加え、非調和部分音と分析帯域端による違いを調べる。
fs=16 kHz、FFT=2048、hop=512、24 frame、24 bins/octとし、分析帯域100–3500 Hzと55–7500 Hzを比較する。
身体はHarmonicとModal、brightness=0.2/0.9、基音=300/900/1800 Hz。
Modalは比率 `[1, 2.3, 3.7, 5.1]`、Harmonicは通常の部分音列を使う。代表条件は同じ振幅・kick・source seedに固定する。
帯域ごとの密度積分質量と、広い分析の3500 Hz超の質量を記録する。
異なる帯域で解析核とピーク選択が変わるため、二つの質量の比を保存率とは解釈しない。

帯域端への逃避が数式上起こり得るかを分離して調べる対照として、3500 Hz以下で0、超過部分で-1の人工Cを使う。
狭い帯域の正規化身体評価は0、広い帯域の評価は超過質量があれば負になる。
これは実音のH/R地形や移動の実証ではなく、分析範囲を失った質量で正規化し直す規則の限界を示す反例である。
1800 Hzの4身体条件では広い帯域に正の超過質量があることを要求する。数値許容は既存の小型演算と同じ。
結果は `BODY_FITNESS_BAND` のJSON行に保存する。

raw Rの部分音対比較では、LUT値を共有する二重和に加え、登録したkernel式をf64で直接評価した二重和とも照合する。
連続式の正規化定数は同じERB刻みと台形端のない登録LUT格子で独立に求め、LUT補間値自体は参照しない。
許容は前記の絶対1e-6または相対5e-4。方向性と中央抑制を含め、対称Sethares式への置換ではない。

## 非線形自己減算の直接対照（追加取得前、2026-09-26）

独立レビューを受け、4静的場面でもmix minus own PCMを他者-only入力へ直接照合する。
先行F1試験と同じPCM・解析後の許容を適用し、4場面のoracle自体を独立検証する。

また、同音のSine二個体、各振幅0.35、436.203064 Hzに最も近い分析bin、24 frame、hold=8000を使う。
混合PCM、単独自己PCM、単独他者PCMを同じ開始状態から別々に解析し、
`max(0, q_mix - q_self)` と `q_other` のERB質量で正規化したL1差を測る。
べき圧縮の非加法性を示す反例としてL1差が0.1を超えることを要求する。
ピーク選択や位相の全条件を代表する試験ではなく、この固定条件での非等価性を示す。
結果は `BODY_FITNESS_DENSITY_SUBTRACTION` として保存する。閾値は結果取得前に固定し、失敗時は緩めない。

## 通常解析設定での追加参照（追加取得前、2026-09-26）

最小試験の8 kHz設定から、`runtime::build_analysis_runtime_core(AppConfig::default(), 48000)` が作る
解析器へ範囲を広げる。現基準のFFT=16384、hop=512、55–8000 Hz、96 bins/oct、右寄せkernel、
平滑化10 ms、既定R/H/C係数を使う。テスト内へ同じ設定を再実装しない。
Sine/Harmonic/Modalを基音440 Hz・振幅0.35で鳴らし、他者なし／440 Hz他者／466 Hz他者の9場面を各64 hop観測する。
各音はhold=48000、attack=5 ms、release=0.5秒、固定source seed、SeqGate=1秒とする。
混合・自己-only・他者-onlyを独立したScheduleRendererで合成し、混合−自己PCMと他者-only解析を比べる。
PCM、解析密度の正規化L1、H/R/Cの許容は先行F1と同じとし、単独除去は厳密なゼロを要求する。
この比較は通常解析設定での数値的な自己減算を調べる。renderer内のsource採取境界、route/tailは先行F1が別に検査しており、
この試験自体ではsource採取や実時間worker配送を新しく検証したとは扱わない。
新しい結果を48 kHzでの実時間資源合格へ読み替えない。

## 固定候補の移動応答アッセイ（追加取得前、2026-09-26）

行動上の自己アンカーへ進む前に、同じ候補分布に対する採否を数値的に分離する。
最初の軽量設定（8 kHz、FFT=1024、24 bpo、24 frame）で、Harmonicのbrightnessを0.1/0.3/0.5/0.7/0.9とする。
現基音は440 Hzに最も近いbin、候補はそこから-120/-60/-24/0/+24/+60/+120セント。
source振幅0.35・hold=8000・従来の代表条件を使い、他者なし／1.5倍の基音のSine他者（振幅0.35）を比較する。
放射RMS対照は、自己PCMを24 frame全体のRMSで測り、合成後に一律のgainを掛けてRMS=0.1へ揃える。
これは励振そのものを変えて音色を再生成する操作とは異なる。通常gainとこのRMS固定を別条件にする。
各候補の身体密度は代表振幅1のToneから実際に再計算し、補正済みscanの横移動では作らない。

旧ExactScan／他者-onlyの2環境と、点／身体平均の2評価を交差させる。
既存の移動費用・音域重力・混雑は加えず、Cの相性による差だけを測る。
6個の移動候補を同確率で提案するMetropolis則を固定し、温度T=0.05/0.2で
`p_k=min(1,exp((score_k-score_current)/T))`、期待受理率=`mean(p_k)`、期待移動距離=`mean(p_k*abs(offset_k))`を計算する。
乱数標本ではなく同じ提案分布の期待値を直接計算するため、seedや乱数分岐の差は入らない。
両環境で7候補すべてを保存し、JSON行 `BODY_FITNESS_FIXED_CANDIDATES` に候補scoreと両温度の期待値を出す。
旧ExactScanは全20場面で有効に呼べること、score／確率が有限かつ確率が0〜1であることを要求する。
他者-only・他者なしでは全候補score=0であり、両温度の期待受理率1、期待距離68セントを要求する。
brightnessに対する単調な増減は合格条件にしない。RMSは絶対誤差1e-6、確率・距離の小型計算は先行参照の許容を使う。

このアッセイは既存PitchCoreの局所窓内のscore上位bin抽出、greedy改善閾値0.1、独自の乱数候補生成を実行しない。
したがって実際のVoiceの提案受理率・移動距離ではなく、固定した提案分布に対する評価差の作用である。
既存PitchCore全体の比較とF3の実時間配送は次の検証として残す。

## 時変Toneとrouteの追加参照（追加取得前、2026-09-26）

軽量8 kHz設定で、Sine/Harmonic/Modalそれぞれの自己sourceと440 HzのSine他者を32 frame観測する。
自己の初期Toneは440 Hz・hold=8000・smoothing=15 msとし、frame 4の途中（+128 sample）に
目標周波数660 Hz・目標振幅0.12へのUpdateを発行する。frame 6にpresentation-onlyのTone 2、
frame 8にhabitatへ出る別身体のTone 3を追加し、frame 10の途中にTone 1のreleaseを発行する。
既存Toneのrouteは発音時のrouteを保持し、後続batchのroute変更が古いToneへ遡及しないことを前提とする。
これは現在のAPIで実際に可能なroute変更であり、鳴っているToneのroutingを書き換える操作は追加しない。

自己-onlyと他者-onlyを別rendererで合成する。実rendererのsource採取PCMが自己-onlyのhabitat PCMと一致し、
混合−採取PCMを解析した結果が他者-only解析と一致することを、先行F1と同じ許容で確認する。
Update後にcurrent frequencyが440〜660 Hzの間に入り、のちに目標へ近づくこと、Tone 2のhabitat routeがfalseで
Tone 1/3はtrueのままであることを検査し、入力が実際に作用したケースだけを数える。
source idは維持し、body generationのruntime配送や途中の解析器登録の検査には読み替えない。

## 既存PitchCoreの単一判断への追加対照（追加取得前、2026-09-26）

固定候補のv4結果を受け、同じ20場面を使って既存の `PitchHillClimbPitchCore::propose_with_scorer` を直接呼ぶ。
局所窓内のscore上位bin抽出、3個のGaussian乱数候補、重複除去、greedy改善閾値0.1、Metropolis選択は実コードを使う。
neighbor step=24セント、大域peak候補0、ratio候補なしとし、移動費用・音域重力・混雑は引き続き加えない。
各環境×点／身体平均について温度0.05/0.2、SmallRngのseedを `0xF17E0001`〜`0xF17E0040` の64個に固定する。
各判断の前にseedから乱数器を初期化し、現在位置・targetは固定基音とする。判断後の乱数状態を次の試行へ持ち越さない。
候補抽出も同じscorerを使うため、評価式を変えた場合に最終候補集合が異なることはこの追加比較の対象に含む。

身体scoreは実際に問い合わせられた基音でToneを再合成して計算する。同じ身体内では基音のf32 bit列をkeyに
代表密度を再利用するが、環境scoreは環境ごとに再計算する。補間や格子丸めは使わない。
80レコードそれぞれに128判断のseed・温度・target・移動距離・salienceと、問い合わせた基音／scoreを保存する。
計10,240判断の全targetが有限・帯域内であること、単独sourceの他者-onlyでは全問い合わせscore=0であることを要求する。
集計は64判断中の移動割合と平均絶対移動距離を記述する。多数試行の推測統計、連続する生態trajectory、
runtime配送の鮮度・費用の試験とは扱わない。移動割合の単調性や特定の差を合格条件にしない。

## 完了の判定

この最小参照の検査を通しても、F1/F2 全体、自己アンカーの行動効果、F3 の費用・鮮度、F4 の選択圧、
F5 の作者採用、F6 の音色遺伝が完了したとは扱わない。新しい入力や数値失敗を根拠に必要な検査を追加し、
実装計画の各終了条件に対応する証拠を別々に記録する。

2026-09-26 独立レビュー注記: 上記の名称は `top_local_peaks` の実装に合わせて正確化した。
関数・候補数・乱数・取得条件の変更はない。v5保存capsuleには取得時点の表記を保持する。
