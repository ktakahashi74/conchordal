# F4の消費境界: 代謝・親選択・出生

日付: 2026-09-26。対象は基準 `06a4772` と、それに基づくF3d隔離実装。
この文書は接続前のコード監査であり、F4の実装・実験結果ではない。
関連: [全体計画](body-aware-fitness-plan.md)、[runtime引継ぎ](body-aware-fitness-runtime-handoff.md)。

## 同じ身体評価を渡す範囲

F2は、代表身体密度で環境の有効C scoreを平均し、その平均をsigmoidへ通したlevelを返す。
成人には本人を除いた環境、新生児には出生前の共有環境を使う。親や死亡した旧個体の音を
新生児の自己音として除去しない。身体密度は現在の励振音量から独立なので、休符中も評価できる。

| 消費箇所 | 現実装 | 身体評価への接続点 |
|---|---|---|
| `Voice::tick_articulation_lifecycle` | 基音のlevelと、点／近似LOOのselection scoreを取得 | 一つの受理済み環境と現在基音の代表身体から得たscore/levelを二つとも渡す |
| `KuramotoCore::process` | levelで基礎消費・attack recharge、設定次第でscoreまたはlevelによる連続recharge | 代謝係数・閾値・clamp・更新順序を維持し、上流入力だけを置き換える |
| `Voice::tick_phonation_into` → `apply_phonation_onset` | 発声gateとGated onsetのrechargeが同じconsonance引数を共有 | gate用の観測と代謝用の身体levelを区別して渡す必要がある |
| `LifeAccumulator::accumulate_tick` | lifecycleが消費したlevelを記録 | 実際に消費した身体levelと、身体評価／fallbackの別を対応づける |
| `weighted_parent_select` | 生存個体の現在energyに比例して親を選ぶ。総和ゼロなら一様。Entrain以外は現在energyを0として集める | 身体評価を受けたenergyをそのまま使う。別の身体scoreを親の重みにさらに掛けない |
| `pick_respawn_candidate` | 16候補を点levelで比較。Randomは正の点scoreを重みに選び、正重みがなければ最大scoreを選ぶ。最後にmin levelで拒否 | 子の確定した身体で全候補を採点し、従来と同じ選択則・最低level判定へ渡す |
| PeakBiased出生 | 点場から候補bin、scene weight、局所探索、最低levelを生成 | 最終判定だけの差し替えでは不足。候補抽出・局所探索も身体scoreへ接続するか、初版では明示的に非対応とする |

`Community::collect_phonation_batches_into` にも基音のlevel読み出しがある。
これは発声gateとonset代謝の双方に届くため、lifecycleだけを差し替えると点評価のrechargeが残る。
一方、GeneratorModelの共有地形予測から作る`extra_gate_gain`は代謝scoreではない。
予測や発声gateまで同時に改変すると、fitnessの作用と発声確率の作用を分離できなくなる。
初回F4の範囲は代謝と出生評価であり、予測・知覚gateの目的関数変更は含めない。
`life::report`の`mean_c_field_score/level`も現状は生存個体の基音における点地形の平均である。
身体評価を代謝に導入しても、この欄を身体fitnessの観測値や選択作用の証拠として読み替えない。

## 成人の一更新で固定する契約

評価する基音は更新時点の身体の周波数であり、移動先targetを代用しない。glide中は両者が異なる。
F3dの候補準備は移動のscorer入力を覆うが、基音がその表に必ず含まれる保証にはならない。
成人代謝用の現在基音も、代表密度の要求集合へ明示的に含める。

scoreとlevelは同じ評価結果から取得する。levelだけを差し替えてselection scoreを旧近似LOOに残さない。
内部attackはlifecycleと同じ制御substepで起こるが、外部Gated onsetは制御substep群の後のphonation hopで起こる。
両者の現在基音・身体世代・環境版・評価時刻をそれぞれ記録し、途中のglideや環境更新を跨いで古い評価を流用しない。
既存のenergy clampとattack順序を維持し、新たな二重rechargeを導入しない。

初回の小さい参照試験は明示offline配送とし、必要な評価が揃わない試行を失敗として記録する。
これを通常演奏での待ち合わせへ持ち込まない。通常runtimeの欠測・期限超過はF3の演奏継続用fallbackを
明示するが、fallbackを含む長期比較には別途、有効評価率と欠測時の代謝の扱いの取得前登録が要る。
特定の身体だけ評価が遅くなる現象を、身体の生存力と解釈しない。

## 子の身体と出生候補を確定する順序

現在は候補周波数と親を選んだ後、runtime idを割り当て、PopulationのtemplateからVoiceを作る。
身体評価では子のrecipeが先に必要になる。Modalの代表Toneはsource idを位相seedにも使うため、
仮のsource idで採点して別idの子を生成するとF1の代表条件が一致しない。

さらに、HarmonicとModalのfactoryは`ModePattern::eval`へ候補基音、周波数空間、
出生時の地形、Voice生成用乱数を渡す。したがって任意のtemplateについて、既存の身体snapshotを
周波数移動しただけで、新しいVoiceをその周波数に生成した身体と一致するとは限らない。
固定ratioの初版と、地形依存のmode patternを明示的に分ける。後者を扱うなら各候補について
実`VoiceSpec::spawn_with_landscape`と同じid・開始frame・metadata・seed・地形から身体を作り、
採用候補のsnapshotと実際の出生snapshotを照合する。候補生成用のspawn乱数とVoice生成用乱数は
別の所有者なので、候補ごとの仮生成で親・候補選択の乱数を進めない。

現行の失敗時の消費も保存する。`spawn_counter`は出生候補を選ぶ前に一回進むが、
runtime idと`next_member_idx`は候補採用後にだけ進む。同期の一機会試験なら未使用idを
読み取って採点し、採用直前の割当値と一致することを検査できる。非同期で先にidを予約する
設計では拒否時の消費規則が変わり得るため、同期試験の成功をそのまま転用しない。

初版の出生接続は次の順序を登録してから実装する。

1. 一つの出生機会・親・子の固定templateを確定する。音色遺伝はここでは追加しない。
2. 子のidentityと代表条件を予約する。出生拒否時のid・spawn counter消費も固定する。
3. 従来と同じ乱数入力・範囲制約で候補を生成し、その子のrecipeで候補密度を準備する。
4. 一つの出生前共有環境で候補を採点し、従来の選択則とmin level閾値を適用する。
5. 採用されたidentity・recipeをそのまま生成へ渡す。準備中に環境やtemplateが失効した場合は、
   過去の出生時刻へ遡って子を追加しない。

初期Population配置、Random／Hereditary respawn、PeakBiased respawnは異なる入口を持つ。
小さい試験を一入口で通した結果を、全出生方式への接続完了とは呼ばない。
特に`respawn_settle_strategy`の補助候補も旧点地形を使うため、共通候補の再採点比較と、
候補生成から身体評価へ変更する比較を分ける必要がある。

## 次の最小検査

最初は成人の一更新を対象とし、固定身体の同じscore/levelが連続代謝とonset rechargeへ届くこと、
旧Cを変更してもその更新の代謝結果が変わらないこと、休符中も入力が有効であることを確認する。
移動先targetとcurrent pitchを異ならせた反例で、代謝がcurrent pitchを使うことを検査する。
旧モードの音声・energy記録を保存し、score閾値付きとlevel依存の両経路を比較する。

出生は別の一機会試験へ分け、子のrecipe・id・親・候補集合・乱数・環境版を保存する。
固定身体の長期選択試験へ進むには、この接続に加え、評価率と欠測時の扱いを先に確定する。
