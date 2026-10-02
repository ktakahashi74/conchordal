# F4c v2: 非ゼロ代謝とHereditary出生の一機会検査

日付: 2026-09-27。実装・取得前登録。旧[F4c登録](body-aware-fitness-f4c-hereditary-registration.md)の未達結果と封印capsuleは保持する。変更は `.worktrees/body-fitness-lifecycle` の試験専用経路に限定する。

旧登録のseed 7、初期energy 0.35/0.65、音源、72 hop代表密度、解析、候補16 slot、sigma 0.03、出生時刻、閾値式を維持する。Sustain templateのenduranceを10秒、recoveryを1秒、dissonance penaltyを1、attack cost/rechargeを0、recharge score boundsをNoneへ取得前に固定する。生存親はretrigger=true、autonomous_attack=falseとする。

各親の実commitをdt=0.01秒で一回実行する。期待energyは身体levelをLとして、f32の順序で `E1 = E0 - 0.05 * (2-L) * 0.01`、`E2 = E1 + L * 0.01`。実energyがこの値と1e-6以内で一致し、更新前とbitで異なり、有限・正・相異であることを必須とする。親選択は更新後energyだけを重みとする。固定seedの一回の抽選結果が別の親へ変わることは要求せず、初期値と更新後の親選択確率も保存して区別する。

身体評価contextを外した旧点C経路を、同じtemplate・初期energy・dtで低点C(level 0.05)と高点C(level 0.95)へ通す。各energyを上式と照合し、両条件のenergy差を要求する。出生閾値0.5では低点Cで拒否、高点Cで出生を要求する。身体経路は点Cの毒入れに不変であることを旧登録どおり確認する。

取得済みの正例contextを複製し、親generation、live template brightness、予約child id、frame、epoch、候補slot順序の六条件を一つずつ破損させる。各条件を通常cleanup_deadから通し、子追加前に拒否され、child id/member indexが消費されず、採点levelを読まないことを確認する。panicによる拒否は試験専用assertionの検査であり、通常runtimeの回復保証ではない。

親抽選前後・候補生成後のRNG probe、身体値、期待/実energy、親確率、六負例、旧点C対照を新しい `target/f4c-v2-offline-evidence/` へ保存する。登録と変更前sourceのhashを実行前に保存し、最初のtargeted実行結果も保持する。失敗後にseed・係数・音源・閾値を調整しない。全cargo testをstdout/stderrとexit code付きで記録し、fmt、Clippyを通す。通常releaseは既存封印版とのbinary照合を行い、不一致なら旧Noneの固定素材回帰を再取得する。

この検査は隔離した一機会接続の検証である。通常runtime、非同期出生、音色遺伝、長期選択、F4全体、実時間資源、作者採用の完了とはしない。
