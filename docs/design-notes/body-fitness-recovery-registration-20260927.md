# 第十四版: 旧世代準備の取消と実時間追随（取得前登録）

第十三版bの非Sine取得は、Harmonicの身体変更frame 300に対する新世代消費がframe 522となり、登録期限frame 432を満たさなかった。旧世代pendingの終了待ちと、新世代132候補×72 hopの冷計算を分けて改善する。第十三版bのsource/binary/rawと不合格判定は保持する。

新しい隔離worktreeは `body-fitness-recovery`、基準は `body-fitness-metabolism/target/integration-source-20260927-v13b/manifest.json` の405ファイル。mainの通常実装には採用しない。取消workerの局所機能試験は先行しており、この登録は未実施の通常action接続と費用取得を固定する。

## 取消と診断

非同期actionのsource退役、Voice出生世代または周波数を除いた代表Recipeの変更を検出したら、保持しているpending serialをworkerへ取消要求する。進行中の一候補72 hopは途中切断せず、候補境界とReady返却前で中断する。古い応答はそのidentityで処理し、後続pendingの解除やVoice決定のcommitへ使わない。既存の厳密な現在環境採点、cache scope、候補・RNG・source照合を維持する。

通常offline modeは決定的な完了待ちを維持し、wall-clock競争による取消は行わない。非同期で一旦受理した取消の完了応答を確認するまで同slotを再投入しない。途中まで完了したcacheは同一source/出生/身体世代/Recipe/解析条件に限って再利用し、新しい身体ではbind時に無効化する。旧密度を新brightnessへ転用しない。

reportにはrequestのserialと身体世代、submit/取消要求/受信のsample時刻、応答状態、候補数・完了候補数、worker計算区間の壁時間を追加する。submitは当該hopの描画後なのでhop終端sample、受信・取消は次のbefore処理のhop先頭sampleで記録する。worker計算時間はCPU時間ではなく、スケジューリング待ちを含む壁時間。offline reportでは壁時間をnullとして決定記録の再現性を保つ。取消要求数と実際の取消応答数を区別する。

## 冷計算の意味不変改善

代表スペクトル解析のscratch再利用と、不要なLandscape再構築を避けるresetを[別登録](body-fitness-spectral-reuse-registration-20260927.md)で扱う。Tone、NSGT、ピーク処理、72 hop、候補集合と順序、係数、score/level、実発音条件を変えない。数値のbit一致を先に検査し、その後に速度を測る。取消やallocation削減だけでframe 432を満たすとは仮定しない。

## 検証と専有取得

機能検査では、実pending中の身体変更と退役の取消、古い応答が新しいpendingを解除しないこと、旧入力からtarget/RNGをcommitしないこと、新世代の再準備・消費、cache上限とscopeを検査する。既存offline actionの繰返しWAV/決定記録一致、provider/代謝/出生の意味も維持する。全suite・format・標準Clippy・全target check後にsourceとrelease test binaryを固定する。

専有取得は第十三版bと同じseed 17、Harmonic 220 Hz/Modal 330 Hz Drone、10秒・938 hop、frame 300 brightness変更、frame 600 control変更、OFF/ON各一回。新しいoffline出生flagも代謝flagもOFF。合否は第十三版bの登録から変更せず、身体回復frame 432、control回復frame 720、継続消費frame 912、全窓2声、underflow 0、hop予算超過0、消費支持age最大4800 samples、cache上限を要求する。背景workerの計算診断と待ち区間を併記する。失敗時はrawと判定を保持し、後から期限・候補数・観測窓を緩めて成功に読み替えない。

同時に進むoffline出生接続は独立flag・独立取得とし、この実時間試験へ混入させない。実device、長期生態、非同期代謝、作者採用は別の未完条件である。
