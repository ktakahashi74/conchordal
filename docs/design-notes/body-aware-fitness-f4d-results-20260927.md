# F4d: settle補助候補付き出生の隔離結果

日付: 2026-09-27。[取得前登録](body-aware-fitness-f4d-registration-20260927.md)に基づく `.worktrees/body-fitness-lifecycle` の `cfg(test)` 一機会検査。Random初回登録のSHA256は `693c0ff01ec314a6c9866e14e3c98d08f47eee16080c0c24d13970e8789cfd75`。Randomのtargeted通過後、Hereditary取得前に追加条件を追記した最終登録のSHA256は `047cf81fd0a2332b9ed3755b2d4ff561949526cd2136d707ae00078b7990cce8`。二版とも `target/f4d-offline-evidence/` に保存した。

Randomではslot 0を従来候補、slot 1–15をConsonance/Densityのsettle補助候補として生成した。16 slotの身体scoreは -0.051421～0.630690。身体scoreの正値を重みにした独立抽選と実選択の周波数bit `1146854584`、終了RNG probe `321572678824776467` が一致した。最低level 0.237156で出生し、0.889632で拒否した。出生時は予約id 2の子の身体・宣言代表recipeと採点時の条件が一致し、id/member index/eventは各一つ進んだ。拒否時はspawn counterだけ進んだ。同じ死亡の再処理では増分なし。

HereditaryではF4c v2と同じ二親の実energy更新を通した。更新後energyは親1が0.355359、親3が0.655280で、親1を抽選した。slot 0を親周波数からsigma 0.03 octaveで生成し、slot 1–15を同じsettle strategyで生成した。身体levelは0.486551～0.784773。最大levelの独立選択と実選択の周波数bit `1130126945`、終了RNG probe `7054783495317049382` が一致した。最低level 0.243275で出生し、0.892387で拒否した。出生した予約id 5の子は親id・世代・身体・宣言代表recipeが採点時の値と一致した。拒否時のid/member index、同じ死亡の再処理も登録どおり。

両方式とも、新しい試験contextを増やさず、F4b/F4cの候補ごとの `Entry`（周波数、実子身体、代表recipe、fitness、identity）と選択・出生後照合を再利用した。候補生成そのものは点地形依存のまま。記録した点score/levelと身体score/levelは別値だが、この一機会だけから長期の選択差は導けない。通常runtimeは身体評価へ常時配線していない。

最初のRandom targetedはRustの借用エラー、最初のHereditary targetedは試験assertionで `SpawnStrategy` に存在しない `PartialEq` を使ったためコンパイル前に失敗した。いずれも試験記法だけ修正し、seed・音源・係数・閾値式は変えなかった。二回目のtargetedは双方通過。全cargo testは2026-09-27 11:36:16 JSTにexit 0。libは1123成功・0失敗・40 ignore、追加二試験の成功行を `test_report.txt` で確認した。`cargo fmt --all`、`cargo clippy -- -D warnings`、`git diff --check`も通過。source・登録・結果JSON・全test reportのhashは `target/f4d-offline-evidence/sha256.txt` に保存した。

通常runtimeへの最小接続差分は、既存 `pick_respawn_candidate` で候補生成後に各候補の実子identity・recipeを固定し、受理済み出生前共有環境の身体score/levelをslot順に選択器へ渡す境界。候補生成が点地形を参照する点と、非同期のid予約・評価期限・失効時の扱いは別途設計・取得が必要。PeakBiasedの全bin・局所探索と初回spawnも未接続。F4全体、音色遺伝、実時間性能、作者採用の完了ではない。
