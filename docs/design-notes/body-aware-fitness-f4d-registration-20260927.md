# F4d: settle補助候補付きRandom出生の一機会登録

日付: 2026-09-27。取得前登録。対象は隔離worktree `.worktrees/body-fitness-lifecycle` の `cfg(test)` 明示offline経路。F4bの固定Harmonic身体・出生前共有環境・seed 7・frame 64・16 slot・代表Tone 72 hop・子id 2・閾値と失効照合を再利用する。F4c v2 capsule、main、統合第八版は変更しない。

既存の `respawn_settle_strategy` はslot 1–15の候補生成に使う。slot 0はRandomの通常候補。今回のsettle strategyは220–880 HzのConsonance/Density、ERB最小距離0、tension 0。候補生成は現行の点地形で行い、その結果をslot順に保持する。点地形への毒入れに対する候補集合の不変性は要求しない。採点前に子のid、template、frame、metadataと実Voiceの身体を確定する。F4bと同じ代表条件で密度を作り、出生前共有環境のF2 score/levelで全16候補を採点する。Randomの既存正score重み選択と最低level判定を保つ。選ばれた子のid、周波数bit、BodySnapshot、宣言代表recipeを採点時の値と照合する。

取得条件: slot 1–15がsettle経路に入ったことを独立再演の候補列とRNG probeで照合する。候補列の少なくとも一つがslot 0と異なり、身体scoreに変動があることを要求する。正score重みからの独立再計算と実選択の周波数bit・RNG probeが一致することを要求する。最低levelを `min(L_i)/2` と `(1+max(L_i))/2` に固定し、前者は出生、後者は拒否を要求する。拒否ではspawn counterのみ進み、runtime idとmember indexは進まない。成功では各一つ進み、event一件。同じ死亡の再処理では再出生しない。

旧点C経路との分離を見るため、同じ候補集合と同じ出生前共有環境で候補の身体scoreと点scoreを記録する。両者の候補順位や実抽選結果の相違は成功条件としない。通常runtimeの点C経路は保持する。settle候補の生成から身体scoreへ変えること、PeakBiased、初回spawn、非同期配送、音色遺伝、長期選択、実時間費用、作者採用は対象外。

変更前SHA256: `respawn.rs` = `1ec6a485f626c77535965088913b2e13fd9c3f36f0e1aaec7e6578dd8cbfde88`、`f4b_offline.rs` = `09a75d90abc7d8d295c4781bb426a0ae1f674dbc32de9cf072c001a7ef59417b`、`f4c_offline.rs` = `e6c1671e9e5af73f4394637d7273b34fdbc6352d6aec176cf5f63c0c11666dda`。登録hashと初回targeted結果を `target/f4d-offline-evidence/` に保存する。変更後は全cargo testのstdout/stderrと同一shellのexit、fmt、標準Clippyを記録する。通常releaseの動作は `cfg(test)` 接続のみのため変えない。

Hereditary追加条件（Random初回targeted後、Hereditary取得前の追記）: F4c v2の実energy更新・親pool・seed 7・子id 5・出生時刻・固定Harmonic templateを維持する。slot 0は親の周波数を中心とする既存sigma 0.03 octaveの候補、slot 1–15は上記Consonance/Density settle strategyから現行点地形で生成する。親抽選、候補16 slot、実子身体採点、最大level選択、最低level、実子生成、RNG probeを独立再演と照合する。候補身体levelに変動があり、`min(L_i)/2`では出生、`(1+max(L_i))/2`では拒否する。採点前の親generation・template・id・frame・epoch・候補slot失効照合は既存F4c v2のhookを使う。既存F4c v2試験はsettleなしで維持する。Hereditary追加取得では点C毒入れの不変性を要求しない。
