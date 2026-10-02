# 提案延期中の実状態変化: 現行の厳格照合結果

2026-09-27。[取得前登録](body-fitness-deferred-state-registration-20260927.md)の四条件を、統合第九版からの検査追加で実行した。初回targetedは1成功・0失敗、exit 0。ログは統合worktreeの `target/deferred-state-validation-20260927/first.log`、終了値は `first.status`。

無変化の正例は古い表を消費した。glideはtargetとRNGを保ったままcurrent pitchが進み、CurrentPitchChangedで拒否された。unison 1→2の実更新ではCaptureのbody generationが1→2へ変化し、BodyGenerationMismatchで拒否された。brightnessの実更新ではgeneration 1を保ち、RecipeMismatchで拒否された。いずれの負例も身体score利用0、表なしのfallback対照とtarget・salience・adaptation・RNG・commit後pitchが一致し、延期時間の残量は0だった。

拒否後の現在状態で表を即時に作り直すと、それぞれ146、143、143回のscore利用で受理が回復した。数値は有限fixtureであり、身体密度の精度や非同期処理の追随を示さない。特に連続glide中は、準備のたびにcurrent pitchが変わるため、現行の厳格照合は永続的に拒否する可能性がある。

この結果は現行照合の動作確認である。候補密度そのものがcurrent pitchへ依存するかは別の依存監査を行う。candidate Hzで生成した代表密度とlive current pitchで計算する移動費用を分けられるなら、厳格照合の条件を変更する前に別登録と直接参照を用意する。元登録・初回結果は変更しない。
