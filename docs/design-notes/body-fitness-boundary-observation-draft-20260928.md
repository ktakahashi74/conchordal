# 非同期出生の高域境界：追加帯域観測の取得前登録草案

2026-09-28。この文書は次の隔離実装・数値取得に先立つ草案であり、まだ取得条件を封印したものではない。対象は元の36条件行列のうち `I-W4-on-a` 1件から得た、封印済みの[高域端点プローブ結果](../../.worktrees/body-fitness-birth-endpoint-probe/docs/design-notes/body-fitness-birth-endpoint-probe-results-20260928.md)に保存された同一PCMである。元36条件行列、690候補の出生判定、通常runtimeのコードと結果は変更しない。追加帯域に右側の谷を観測できるかだけを調べる。新しい出生、身体採点、確率選択、30秒の期限達成はこの単位の判定対象外とする。

## 既知の事実と仮説

元の解析空間は `Log2Space(55, 8000, 96)`、690 bin。実際の予約jobのticket 1/2から候補index `288, 686, 687, 688, 689` を採り、各72 frame・512 sampleのPCM、NSGT powerなどを保存した。元の[取得plan](../../.worktrees/body-fitness-birth-endpoint-probe/target/endpoint-probe-acquisition-v1/plan.json)のSHA-256は `e60e68b1c443f1400b7045c713ac876c847cf03c08eade33d07f6f27b857236f`、[probe manifest](../../.worktrees/body-fitness-birth-endpoint-probe/target/endpoint-probe-acquisition-v1/probe/manifest.json)は `79816e1aab5f4f0636f79ae9cc3b6dfae300af2805ed574560fa288b56950fab`、[追加prominence監査](../../.worktrees/body-fitness-birth-endpoint-probe/target/endpoint-probe-prominence-v1/audit-v1.json)は `b498034e1ed3d677c834102b08887fef83a93e405eab83ec92744a7b163ab112`。取得前にこれらとPCM 10本の実物SHA・長さ・frame順を再照合し、別planへ固定する。

追加監査は既に、index 687/688の対象binが全frameで旧帯域の右端まで探索され、右側の最小値によりprominenceが10 dB未満となることを確認した。687は3.44002–4.42113 dB、688は0.84218–1.10675 dB。index 689は端binなので現在の内側候補に入らない。[封印済み設計比較](../../.worktrees/body-fitness-birth-endpoint-probe/target/boundary-semantics-next-design.md)の「実数値・谷位置は未計算」という記述は追加監査より前の時点を示す。今回の問いは、その後に判明した旧帯域の打切り位置より上まで**実際に観測範囲を広げたとき**、右側の谷とprominenceの判定がどうなるかである。

## 固定入力と二つの解析

元manifestのticket 1/2 × 上記5候補、計10本の保存済みPCMをそのまま使う。各queryは72 frame × 512 sample = 36,864 sample、全10本で368,640 sample。成功しそうな候補・frameだけを選ばず、計720 frameすべてを処理する。各候補のRecipe、body、初期状態とframe順は元probeの固定値を参照し、新Toneを鳴らさない。旧690点のpower/peak/主観強度を新しい結果で上書きしない。

新しい観測空間は事前に `Log2Space(55, 16000, 96)` と固定する。上限は旧8000 Hzの1 octave上、48 kHz標本化のNyquist 24 kHz未満である。解析は旧probeと同じNSGT設定、すなわち標本化48,000 Hz、hop 512、`nfft_override=16384`、`KernelAlign::Right`、`PowerMode::Coherent`、窓capなし、同じリセットと72 frame順を用いる。旧probeの `RtNsgtKernelLog2::new` が使う `RtConfig::default()` の全field、`tau_min=0.005` 秒、`tau_max=0.020` 秒、`f_ref=200.0` Hzも固定する。取得前planには各fieldの値とf32 bits、NSGT設定、リセット順、frame順を記録する。これらは旧prefix powerのbit照合に必要な前提であり、値が違えば同一条件の比較と扱わない。公開候補は引き続き旧690点だけとし、新空間の追加binを候補・出生抽選へ流さない。新しい `fmax` で得るbin数と中心列の実値は取得前planに保存する。旧index `0..689` の中心Hz/log2のf32 bitsが新空間の同indexと一致することを必須とする。

第一の**旧基準に接続する対照**では、公開帯域のdensity、ERB幅 `du`、最大densityに基づく床を元rawと[旧prominence監査登録](../../.worktrees/body-fitness-birth-endpoint-probe/target/endpoint-probe-prominence-v1/registration.md)から再構成する。新たに観測したguard binだけは拡張空間のpowerとERB幅でdensityを計算し、旧bin `0..689` に続ける。元の内側候補集合・旧閾値・左側最小値が監査済み値と一致することを先に要求する。これは境界の先を観測するための**混合診断列**であり、新しいproduction密度や身体採点として用いない。全5候補の全frameに同じ手順を適用する。

第二の**拡張空間そのものの観測**では、全binのERB幅を新空間で計算し、全binのpowerからdensityを作る。相対床は全拡張binの最大density、絶対床は従来どおり `1e-6`、相対床は `-50 dB`、prominenceは `10 dB` とする。旧index `0..689` の候補性、左右の谷、prominenceを再計算し、新たな候補とguard側の強度も記録する。`du` は `erb_grid` の端点式に依存するため、旧最終bin 689 は拡張時に内側binとなり `du` が変わる。中心周波数がbit一致しても、そのdensity、床、peak質量、主観強度のbit一致を推定しない。全域最大値や後段のpeak選別も変わり得る。二つの解析の値を混ぜて「旧結果の再現」と呼ばない。

両解析とも、保存済みPCMから再計算した新NSGT powerの旧prefix `0..689` を、元の各frameのNSGT power f32 bitsと照合する。prefix中心またはpowerに不一致があれば、全rawと不一致位置・bits・statusを保全し、**同じ信号の帯域拡張という比較を停止して未確定**とする。許容誤差でbit不一致を消さず、合格候補だけを抽出しない。既存の抽出済みPCM、旧NSGT設定、frame順が本当に揃っているかを先に検査する。

## 記録と判定

各ticket・query・frameについて、入力/出力のSHA-256、旧/新spaceの全中心bitsと `du` bits、旧prefix power照合、追加binのpowerとdensityを保存する。全720 frameのrawと局所極大・床通過候補の集合を欠落なく保持する。旧基準対照と拡張空間観測のそれぞれで、最大density・床、**閾値を通過した内側候補全件**の左右探索順・左右最小値とbin・停止bin・旧末端689と新末端への到達・prominence f32値・10 dB比較結果を記録する。加えて、各queryの固定元index `288/686/687/688/689` も同じ詳細を記録し、候補でなければ端bin除外、床未達、左右隣接条件不成立などの理由を示す。端bin0など、調査対象外の全binへ架空の左右探索を割り当てない。境界近傍は旧監査と同じ±`2e-5` dBを「数値未解決」として別計数し、閾値を緩めない。どの候補も有限性と正負ゼロを原f32 bitsで保持する。

「右側の谷を観測」は、旧末端より上の内側guard binに候補より低い局所最小があり、その後の観測binに上昇がある場合に限る。現在のprominence走査が候補より大きいbinで停止したか、最後のguard binまで到達したかも独立に記録する。末端まで達し、谷の後の上昇も確認できない場合は右打切り。guard内に谷があっても停止条件を満たさない場合は「谷は観測、走査終端は右打切り」と区別する。谷が旧帯域内にある場合や最終peak中心がguard内だけにある場合も別分類とする。全720 frameと各判定の件数を表示し、欠落・NaN/Inf・入力不一致は失敗原本として残す。

観測した谷、10 dB通過、最終peak支持、公開690 bin内の正の主観強度massは異なる段階である。今回の主判定は前二者まで。peak支持や公開帯域だけのmassを補助に計算する場合は、guard側のpower/massを公開帯域へ移さず、別schema・別分母に記録する。guard内のpeakが現れても公開候補の身体密度成立とは扱わない。通常workerの全690候補準備、30秒内の3出生、資源上限は測定しない。guard幅を今回の結果に合わせて調整せず、不成立・右打切りもそのまま結果とする。

取得に進む前に、この草案とは別に実装source、解析器と合成反例、固定plan、旧raw実物のhash、出力schema、失敗時partial rawの扱いを封印する。取得後に幅・閾値・候補・期間を変更した版は別登録・別結果とする。
