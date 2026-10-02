# 部分音群モデルv2：隔離試作の取得前契約

2026-09-29。[草案](body-fitness-partial-groups-draft-20260929.md)から選ぶ一案。v1の意味・費用不合格を保持し、本書の別モデルを隔離実装する。runtime接続は本書の進行条件を通ってから別途検査する。旧72-frameの再現実装ではない。

## 1. 身体と代表状態

実子のBodySnapshot、source ID、fs、hopを入力とする。部分音比・gain・unisonの分割・detune・Modalのt60/in_gainとamp_scaleの扱いはv1と同じ実backend由来。Harmonicのk番目の基礎partialには、先の単一介入で定義した `D_k=mean_[0,T](0.3+0.7 exp(-t/0.5))^k` を掛ける。Modalはv1の解析的な平均二乗減衰を使う。T=72*hop/fs、最初の取得は48 kHz・hop 512・T=0.768秒。振幅二乗は代理powerであり、PCM/NSGTの絶対powerと同一とはしない。

Harmonicのmotionは代表振幅1の共通周波数offset `f0*delta` とする。実backendの8 sample周期のLFO/PinkNoise状態を一度だけ進め、各hopのsample `j*hop+floor(hop/2)` を実際に処理するときのdeltaを72個採る。source IDは `modal_phase_seed(id,0,0)` を経由する実seed規則に従う。motion=0、Sine、Modalはdelta=0の1点。72個をsortし、個数とdeltaのprefix sumを保持する。候補ごとに72時点の状態更新を再実行しない。

Harmonicの平均減衰とmotionは分離した代理モデルである。実ADSR/modulator、drive、振幅に依存するjitter scale、dampingとmotionの時間相関、lane間の位相干渉、NSGT窓・peak検出・解析平滑を再現しない。入力にこれらがあるfixtureも除外せず、旧参照との差を保存する。

## 2. 名目周波数での部分音群

各候補f0について、名目周波数 `f_l=f0*ratio_l` を作る。Sine/HarmonicのNyquist maskとModalの `[1,0.49*fs]` clampは実backendと同じ。次に名目周波数が分析帯域外のlaneを除外し、proxy powerを `nominal_out_of_band_power` に保存する。端へ移さない。以後の群形成とmotion平均にはこの除外分を再投入しない。

同一Hzのpowerを先に合算する。残った点をpower降順、同値時Hz昇順として、互いに0.2 ERB以上離れるanchorを最大32個選ぶ。すべての残存powerをERB距離最近のanchorへ割り当てる。同距離は低いHzのanchorを選ぶ。群powerとpower重み付きERB和を保持する。

群powerが最大群の `10^(-50/10)` 未満、または全群powerの5%未満なら、その群のpowerとERB和を最寄りの保持anchorへ移す。距離は元のanchor間で比較し、再配分途中にanchor位置を更新しない。全群が落ちる場合は最大power、同値時最低Hzの1群を保持する。最後に `u_g=sum(p*u)/P_g`、`f_g=erb_to_hz(u_g)` を求める。0.2、32、−50 dB、5%は既存peak設定由来だが、この群形成は新規則であり旧peak検出器ではない。絶対density floorとprominence判定は持ち込まない。

群形成は名目音高で行い、motionでgroupの分離を再判定しない。これは計算を有限にするモデル上の近似。名目帯域外のlaneがmotionで入る寄与も本版では含めない。一方、名目帯域内の群がmotionで外れる寄与は次節で除く。この非対称性を隠さず、帯域端の反例を別記する。

## 3. 群を聴感massへ変換し、motion分布で評価する

環境ごとに一度、Log2Space中心 `f_b` で `W_b=A_power(f_b)^e`、`V_b=W_b*C_eff_b` を準備する。eとref_powerのsanitize規則はv1と同じ。Cの全binの有限性、duの正値、representationの有限性、named scan長をこの境界で検査する。ここは数値試作の境界であり、実runtimeのepoch照合まで完了したとは扱わない。

候補ごとの各群について `M_g=(P_g/ref_power)^e`、`f_gj=f_g+f0*delta_j` とする。帯域内のf_gjだけを採り、隣接するLog2Space中心間でWとVを**Hzに対して線形補間**する。bin区間はlog2座標で求める。線形Hzの配列indexでscanを読まない。最初と最後の中心からfmin/fmaxまでの短い端区間は、その端bin値を保つ。範囲外のf_gjには端binを使わない。

```text
mass = sum_g M_g * mean_j[ I_inband(f_gj) W(f_gj) ]
numerator = sum_g M_g * mean_j[ I_inband(f_gj) V(f_gj) ]
score = numerator / mass
level = sigmoid(beta * (score-theta))
```

これは群powerの一度の指数変換後に周波数方向へmassを分配する定義である。massを2 binへ分配してからpower指数を掛けない。診断scanは同じ補間係数とW_bからbin massを作り、duで割る。全scanはLog2Space長を守る。

実装では、同じ補間区間に入るdeltaの個数と総和をprefixから取り出す。W/Vはその区間で `a+b*f` なので、72回の内挿の平均を個数と総和から計算できる。これは上式に対する代数的な短縮であり、さらに別の時間近似を加えない。独立参照は72点を直接列挙する。

motionで帯域外になった群の代理powerを `motion_out_of_band_power=sum_g P_g*(1-inband_probability_g)` として別記する。これは群中心を揺らすモデルの診断量であり、全laneの実音の帯域外powerとは呼ばない。名目帯域外量との区別を維持する。有効massゼロ、不正入力、容量超過は明示失敗とし、古いscratchや前候補の結果を返さない。

## 4. 仕事量と実装範囲

v1と同じB<=2048、N<=2048、J<=256、K<=64、U<=9、L<=576、同hop16出生・共存32 Voiceを対象目標とする。対応family、候補、対象負荷を縮めない。取得時の候補数は690＋21、最大容量は別観察。

身体ごとにlane、power順位、motion profileを一度準備する。候補ごとのscratchは非ゼロlaneと最大32群だけを扱い、通常のscore計算ではB長の配列を消去・構築・積分しない。環境のW/V検査・準備は同じbatchで共有する。群選択は最大32anchorとの照合で上界を持つ。mass割当は周波数順の点とanchorの走査で行える。motion評価は各群が跨ぐbin区間数Sとprofile長Qに対するO(S log Q)。Q<=72、S<=Bなので上界を隠さず記録する。診断scanの構築費用は通常のscore経路と分ける。

実装はcfg(test)の新moduleに置き、v1と凍結binaryを保持する。候補依存Landscape身体は候補ごとに同じ実生成規則を使う。候補ごとのVoice全体生成を減らす場合は身体・乱数の一致を別に検証し、それまでは省略した費用で一般出生成立を主張しない。

## 5. 取得前の判定

新式の独立f64参照はlaneから群を作り、motionの全標本を直接列挙する。正規化bin mass L1とmass相対差は1e-5以下、score/level絶対差1e-6以下。prefix集計と直接列挙を、負のoffset、同値offset、fmin/fmax、最後のbin中心より上の範囲で比較する。群のtie、0.2 ERB境界、弱群再配分、同一Hz power分割、振幅scale、unison幅ゼロ近傍、不正Cがbody支持外にある場合、失敗後の状態を検査する。

保存済みの13身体×2基音×7音高×4環境、全728入力と旧72-frame結果を再利用する。新旧比較のscore 0.025、level 0.0125、旧gap 0.1以上の順位逆転0は維持する。新式の内部検査が成功しても旧比較の失敗を救済しない。保存fixtureは設計診断にも使ったため、独立したholdoutとは呼ばない。名目帯域の処理とmotionの非対称性、群の境界を反例として報告する。

費用はreleaseで固定身体12条件、Landscape4条件、Harmonic motion=0.9のK16/U1・K64/U9の各1/16体（4条件）、最大容量の別観察1条件。各5回の最大値を保存する。固定身体とmotionには非平坦な `C_b=sin(0.037*b)` を使い、v1の平坦Cとの厳密な速度比は主張しない。1体1 ms、16体8 msの上限は維持する。cold身体準備・scratch・共通環境準備を時計に含め、16体で環境準備を共有する。診断scanは作らない。全実子生成を含むLandscape費用を別記する。最大容量には引き続き合否閾値を後付けしない。

全条件の結果を保存し、通ったfamilyだけを本番へ接続しない。内部式、旧比較、費用の三つを通るまでは同hop出生の統合に進まない。

## 6. 取得器の身体種別訂正（費用取得前）

最初のsemantic取得は、取得器が `landscape_peaks` をModalと仮定したassertで停止した。凍結したv1入力では、このcaseの56行すべてがHarmonicである。原因はv1の単独取得器が通常runtimeの `life::modal::register_modal()` を呼ばず、Modal指定がHarmonicへfallbackする生成経路に入ったことだった。保存728入力と旧結果を変更せず、v2比較は保存BodySnapshotの実際のkindを使う。数式・誤差上限・順位条件も変更しない。途中取得と修正前binaryは保持する。

v2の費用はまだ取得していない。LandscapePeaksのModalという意図した条件を測るため、取得器では通常runtimeと同じModal factory登録を計時前に行い、各候補の生成結果が指定kindと一致することをassertする。起動時のfactory登録は出生計時へ含めず、候補ごとの実Voice生成は引き続き含める。要求kindと実kindを記録し、Harmonic fallbackをModal負荷として数えない。この訂正により、v1のLandscapePeaks費用との同一入力比較は成立しない。
