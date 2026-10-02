# 直接部分音モデル v1：隔離試作の取得前契約

2026-09-29。[再出発方針](body-fitness-runtime-restart-20260929.md)に従う一案。ユーザーは方針と作業継続を承認した。本書は試作の定義と事前判定であり、本番採用や数値・資源の合格ではない。試作場所は `.worktrees/body-fitness-direct-model/`。time4版の固定コピーを基にし、mainの並行変更には触れない。入力監査は[身体](../../target/body-fitness-direct-model-20260929/model-input-audit.md)、[検証範囲](../../target/body-fitness-direct-model-20260929/validation-scope-audit.md)、コピー元の全ファイルは[目録](../../target/body-fitness-direct-model-20260929/base-source-manifest.json)に保持する。

## 1. 入力と代表状態

実子の確定 `BodySnapshot`、候補Hz、sample rate、Log2Space、ERB幅、既存のA-weighting power gain・loudness exponent・reference powerを入力とする。Sine/Harmonic/Modalの全familyを扱う。Landscape依存patternはその時点で生成した実子のratiosを使う。別の環境やseedで代用品を作らない。固定snapshotを別候補へ移す際のratioと、出生時にpatternを再生成する際のratioは区別し、実消費者への接続前に採点と実子の一致を検査する。

Sineは1 lane。Harmonicは実backendと同じ既定部分音数、確定ratio優先、brightness、inharmonic、spread/unisonのgain分割・detuneを使う。代表damping energyは1。Modalは既存 `modal_modes_from_ratios` によるratio、gain、in_gain、t60を使う。`amp_scale` は実Toneと同じく有限値を[0,1]へclampして振幅係数に含める。loudness exponentとreference powerの下限も既存front-endと同じ0.01と1e-12とする。

Harmonic/Sineの高域maskとModalの `[1, 0.49*fs]` への周波数clampは実backendに合わせる。分析帯域外は端のbinへ移さず、そのpowerを帯域外量として記録する。全帯域外なら失敗状態を返す。

代表時間は `T=72*hop/fs`。最初の比較は通常の48 kHz・hop 512、T=0.768秒。Modalのmodeごとの自由減衰を以下で近似する。

```text
x_l = 2 ln(1000) T / t60_l
D_l = -expm1(-x_l) / x_l
a_l = amp_scale * gain_l * in_gain_l   (Modal)
a_l = amp_scale * gain_l               (Sine/Harmonic)
p_l = a_l^2 D_l                        (Modal)
p_l = a_l^2                            (Sine/Harmonic)
```

振幅二乗の代理量であり、PCM/NSGTの絶対powerとして校正された量ではない。全lane共通の1/2は正規化scoreから相殺されるため省く。t60は実mode compilerの範囲に合わせる。

省くものは、lane間の位相干渉、resonatorの入力位相と過渡、ADSR/render modulator、Harmonicの時変damping・motion、pitch/amp smoothing、NSGTの窓と漏れ、peak検出、解析器の時間平滑である。入力にmotion等があっても対応済みの時間変化とは呼ばない。比較にこれらの反例を含め、誤差で採否を判断する。新式では全lane共通の正の振幅係数はscoreから相殺されるが、旧参照でADSRが一般に相殺されるとは主張しない。

## 2. 分布と評価式

帯域内周波数だけ `Log2Space::index_of_freq` で最近接binに写像する。同binのpowerを先に合算する。二つのbinへpowerを分割してから指数を掛ける方式は使わない。

```text
P_b = sum(p_l where bin(f_l) == b)
q_b = ((P_b * A_power(center_hz_b)) / ref_power)^exponent / du_b
m = sum(q_b * du_b)
score = sum(q_b * du_b * C_eff_b) / m
level = sigmoid(beta * (score - theta))
```

qは新モデルの「聴感補正したpower代理密度」であり、旧 `subjective_intensity` と同値ではない。qの単位はreference powerで規格化した代理量の指数/ERB、mはそのERB積分。分布の比較には正規化bin質量を使い、mの旧参照との差は絶対powerの誤差と解釈しない。

候補集合、最新環境、占有、spacing、選択式、乱数は最初の比較で固定する。環境自己除去のPCM→解析経路を変更しない。全 `_scan` の長さはnamed hard assertで確認する。無効値、容量超過、質量ゼロは成功値0と区別する。

この純粋な計算試作ではcallerが同一空間のCとERB幅を渡す。長さの検査だけでは空間・epochの一致は証明できないため、実fixtureは同じ解析入力から得た空間・幅・環境を照合する。実消費者への接続前には、Landscape・epoch・spaceを組にして確認する境界が別途必要である。失敗後に前回や途中の密度を返さない。`NoInBandMass`では全laneの走査を終えた当該候補の帯域外powerのみ診断値として保持し、無効入力等の失敗ではその診断値もクリアする。

## 3. 仕事量・負荷・失敗状態

試作の入力上限は分析bin B=2048、候補N=2048、追加実Hz J=256、基礎ratio数K=64、active unison U=9、L=K*U=576 lane。同hop出生16、共存Voice32を対象負荷の目標とする。これは無制限のcountを受け入れる保証ではなく、既存sampleの18共存・研究sampleの16同時出生を含む試作範囲。範囲を超える入力は明示拒否し、切り詰めない。通常sampleの適合も別に確認する。

一身体の準備はO(L)、候補分布と採点はまずO((N+J)(L+B))、作業保持はO(L+B+N+J)。N枚の密度表を保持しない。lane準備・scratch容量確保と、全候補・局所Hzの採点を含めて測る。実子生成、Landscape mode生成、環境更新、既存の生態・音声処理の費用は別に加算する必要がある。

最初の費用screenはrelease、48 kHz、55–8000 Hz、96 bins/oct、全690候補と21個の非格子候補を使う。Sine/Harmonic/Modal、K=16/U=1とK=64/U=9を含め、cold一身体とcold 16身体を測る。各条件は5回、最大値を保存。準備込み一身体1 ms以下、16身体batch 8 ms以下を試作の進行条件とする。8 msは1 hopの10.667 msから他処理へ最低2.667 msを残す設計上限であり、実際の残余予算を測った値ではない。これを通っても全hop・共存32 Voice・deviceの受入ではない。最大B/N/Jの仕事量は別途測り、690条件の結果で一般化しない。

未準備・cache missはこの計算経路で処理し、cache命中を成立条件にしない。試作は純粋な計算だけでVoice/ID/RNGを変更しない。接続段階では一つの出生をprepare→validate→select→commitとし、失敗した子のID/counter/Populationを部分更新しない。同hop複数出生は従来の逐次順とし、先に成功した子は後続失敗で巻き戻さず、各子の成功/失敗を記録する。失敗を点評価、古い環境、秒単位の延期に隠さない。選択に使った身体と実子が一致しない場合も失敗とする。

## 4. 事前の数値・意味判定

新式の独立f64計算とRustの正規化bin質量L1・m相対差は1e-5以下、score/level絶対差は1e-6以下。独立側は新Rust評価器を呼ばない。単一bin、同bin重複、非均一ERB幅、正の振幅scale、不正入力、全帯域外、Modalの高域clamp、unison、非格子Hzを含める。独立式の一致を旧モデルとの一致と数えない。

旧72-frame参照との小比較では、score最大絶対差0.025以下、level最大絶対差0.0125以下、旧score差が0.1以上の候補対の順位逆転0を進行条件とする。0.025は既存PitchCoreのgreedy改善閾値0.1の1/4で、既定beta=2のsigmoid微分上限0.5からlevel上限を定めた。これは作者受入や知覚弁別閾ではない。候補の選択確率・選択差と正規化分布L1も全件保存する。分布L1だけでは不合格/合格を決めず、行動量の条件の代用にしない。

比較身体はSine、暗/明Harmonic、非調和Harmonic、暗/明Modal、spread/unison付きHarmonicとModal、LandscapeDensity/Peaksから得た実ratio、motion有り、異なるADSR/modulatorを含む。同一身体の候補は基準440 Hzの -120/-60/-24/0/+24/+60/+120 centを固定。帯域端は基音1800 Hzの同集合で別比較する。環境は共有入力から得た他者-onlyの無音、Sine 440 Hz、Sine 660 Hz、Harmonic 330 Hzに固定する。純粋な式の反例用人工Cと実音由来Cは結果を分ける。

LandscapeDensity/Peaksの比較では、候補Hzごとに同じID・community seed・seed frame・Landscape・specから実子を生成し、その同じRecipeを新式と旧参照に渡す。440 Hzで得たratio列を全候補に使い回す固定snapshot比較とは分ける。旧time4取得はmode patternを含まないので、新fixtureとして実際の生成経路を通す。対応する冷状態費用には、候補ごとの身体生成・pattern評価も含めて別記する。

全対象の合否を保存し、一部familyだけの成功を全体合格としない。失敗した場合は式の実装誤りとモデル差を分け、閾値を緩めず失敗理由を保存する。新しい仮説が必要なら別契約にし、同じ取得を再分類しない。小比較と費用screenを通過した場合にだけ同hop出生の隔離接続へ進む。旧36条件の一括再取得、汎用cache、別の候補探索、I11チューニングはこの単位に含めない。
