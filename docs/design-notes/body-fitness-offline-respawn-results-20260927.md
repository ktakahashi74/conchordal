# 通常offlineの親付きrespawn: 第十七版fの結果

2026-09-27。[事前登録](body-fitness-offline-respawn-registration-20260927.md)に固定したseed 7、48 kHz、512 samples/hop、同一PopulationのHarmonic Entrain founder 3声、380–520 Hz、背景死亡率0.5/秒、終了時刻1.6秒を変更せず取得した。OFFは身体respawnと身体代謝の両flagをfalse、ONは両方trueとし、それぞれ2回実行した。高閾値1の拒否対照を1回実行した。これは身体出生と身体代謝を同時に切り替えた比較であり、各効果を単独には分離しない。

**この文書の一次記録は最終sourceの全suite中に生成した取得rawである。全suiteは1322成功・0失敗・48 ignoreで終了した。** [summary](/home/shafi/lwrk/conchordal/target/body-fitness-respawn-build-20260927/runtime-respawn-evidence/registered-4171-1790496039423805587/summary.json)と[全raw](/home/shafi/lwrk/conchordal/target/body-fitness-respawn-build-20260927/runtime-respawn-evidence/registered-4171-1790496039423805587/)に、各実行のscene、config、WAV、JSONL、stdout/stderr、終了コードとhashを保存した。v17fの[source manifest](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-respawn/target/integration-source-20260927-v17f/manifest.json)自体のSHA-256は `22791e93aac37af82c606877d4fb5f17f68079e8fcb223aa46192cc128a58ae9`。[取得時のsource snapshot](/home/shafi/lwrk/conchordal/target/body-fitness-respawn-build-20260927/runtime-respawn-evidence/registered-4171-1790496039423805587/source-files.json)は248ファイルであり、manifestの全414ファイルを取得ごとに個別記録したという意味ではない。使用した `conchordal-render` のSHA-256は `2fd7a5fb59486d791f4501074b792a60138ab1593567924f26682bce98273cec`。各取得の[binary・scene・config対応](/home/shafi/lwrk/conchordal/target/body-fitness-respawn-build-20260927/runtime-respawn-evidence/registered-4171-1790496039423805587/on-a.acquisition.json)も保存した。

## 最初の死亡、親選択、身体候補

RNG-onlyの事前予測どおり、最初の単独背景死亡はframe `143`、Voice `3`、出生決定sample `73216` に起きた。OFF・ONともに生存親poolはVoice `1`（約380 Hz）とVoice `2`（約450 Hz）で、選択親はVoice `1`。親抽選seed `3303204862164289706`、選択前probe `1036442626469085964`、親選択後probe `17337611181627459022` は一致し、登録した独立SmallRngによる親重み付き抽選とも照合した。

| 親 | ON: hop開始energy | ON: 更新後・親抽選energy | OFF: 親抽選energy |
| --- | ---: | ---: | ---: |
| Voice 1 | 0.3146169186 | 0.3098237514 | 0.34690475 |
| Voice 2 | 0.3148269057 | 0.3100332618 | 0.35487545 |

ONの重みはphonation・lifecycle後の実energyであり、hop開始値ではない。選択親Voice 1のOFF/ON差は約 `0.037081`、Voice 2は約 `0.044842` で、登録閾値 `1e-6` を超えた。ONの診断では親選択から最終候補選択までのRNG probeも記録した。

ONは380–520 HzのLog2Space全 `44` binを予定子のHarmonic bodyで評価し、`16` peak候補からリストindex `14`、絶対bin `289` を選んだ。その中心は `443.1885070800781 Hz`、身体score `0.0434570387`、level `0.5217148662`。既存の局所探索で `21` 点を訪れ、最終 `448.3377380371094 Hz`、身体score `0.6422325373`、level `0.7832088470` を得た。全bin、peak、局所候補と最終選択は[ON report](/home/shafi/lwrk/conchordal/target/body-fitness-respawn-build-20260927/runtime-respawn-evidence/registered-4171-1790496039423805587/on-a.jsonl)に残した。integration試験は全binの座標と記録されたpeakの対応、各peakの重みを検算し、**記録されたpeakリストを入力として**独立SmallRngで重み付き抽選を再現した。局所探索も記録された各点のscoreから訪問Hz・最良点を検算した。peakリスト自体を全binから独立に再抽出した試験ではなく、局所点のscoreを通常renderer環境で独立計算した試験でもない。別の局所fixtureでは、別Tone・別Analysisで構成した環境に対し、1 binと選択最終Hzの72hop身体score/levelを直接積分で照合した。この数値対照は通常renderと同じ環境の値ではない。通常renderの候補表では、例えばbin `295`、`462.8101501464844 Hz` の身体scoreは `-0.0477073900`、従来の点scoreは `-0.4028300941` で、差は約 `0.355123`。身体候補と点候補が数値上異なることを確認した。全連続周波数を直接最適化したという意味ではない。

OFFの最終子Hzは `448.98578`、ONは `448.3377380371094`。この差は観測結果であり、事前合否条件としてmode間のHz差を要求したものではない。ONの予定子と実子は同じID `4`、Population `1`、member index `3`、親ID `1`、**系譜generation `1`**、Harmonic body、最終Hz、source ID `4`・**出生時body generation `0`**・32 byte Recipe hashで一致した。系譜generationとRecipeのbody generationは別の識別軸である。成功時のcounterは `spawn_counter 1→2`、`next_runtime_id 4→5`、`next_member_idx 3→4`。cleanup直後のsource集合はID `1,2,3,4` で最大4 sourceを記録した。

## 出生hopと翌hop

[出生遷移report](/home/shafi/lwrk/conchordal/target/body-fitness-respawn-build-20260927/runtime-respawn-evidence/registered-4171-1790496039423805587/on-a.jsonl)では、frame `143` のobserver入力batchは出生前の3 source。新子ID `4`、出生sample `73216` の代謝receiptは `null`、状態は `not_evaluated_birth_hop`、energyは `1.0`。既存Voice `1,2,3` にはこのhopの `source_removed` receiptがある。出生時の子の自己PCMは `512` samples、絶対値和 `0`、全sampleゼロ。これはphonation収集後に生まれた子へ未存在の自己音を割り当てない境界を実測した。

次のframe `144` のobserver入力batchは4 source。report上の現存Voiceは `1,2,4` で、死亡Voice `3` は退役した。子の初回receiptは `source_removed`、source identityは `(id=4, generation=1, birth_sample=73216)`、decision sampleとsupport終端は `73728`、epochは `0`、空間は690 bins・55–8000 Hz・96 bins/oct。子のenergyは `0.9955024719` に更新され、出生時の `1.0` より低い。receiptのbody generation `8` は代表Recipe識別の更新値であり、8種類の物理的なBodyが生じたという測定ではない。出生hopに退役Voiceがobserverへ残り、次hopで集合が縮むことはこのraw範囲で確認した。[Finish診断](/home/shafi/lwrk/conchordal/target/body-fitness-respawn-build-20260927/runtime-respawn-evidence/registered-4171-1790496039423805587/on-a.jsonl)はframe `150`・sample `76800` に、初回respawn機会の成立と子ID `4` の翌hop受領完了を記録した。事前RNG-only計算で予測した次の背景死亡frame `154` より前に終了した。混合音へのtail寄与や長期退役の一般性まで、この取得から主張しない。

## 拒否、再現性、検証境界

最低levelを `1` とした対照は同じframe `143` の選択後、最終level `0.7832088470` に対して `min_level` で拒否された。[拒否report](/home/shafi/lwrk/conchordal/target/body-fitness-respawn-build-20260927/runtime-respawn-evidence/registered-4171-1790496039423805587/reject-min-level.jsonl)の `actual_child` はnullで、respawn eventもない。`spawn_counter` は `1→2`、`next_runtime_id` は `4`、`next_member_idx` は `3` のまま。子IDやmemberを消費しなかった。

各mode内の2回はWAVのSHA-256と、`spawn`、`respawn`、`death`、`population_step`、身体respawn・代謝・遷移・ゼロPCMの**対象JSON record**が一致した。OFF WAVは `f74c5512ba32c35f596ae7e293fdc2a4c9e11bc1cb881096d0bcc2d080f847f8`、ON WAVは `4f6bde7d1b43292aca61ae652b45c63b6d38aa0a4574024f119923617ad29658`。mode間のWAV差は今回の観測値であり、原因を出生単独と代謝単独へ分解できない。補助的な時刻等を含むJSONLファイル全体のbyte hashは同mode内でも異なるため、全JSONLのbyte一致とは呼ばない。OFFでは身体fitness診断が現れず、既定falseの通常点評価経路を通った。

[シナリオ負例raw](/home/shafi/lwrk/conchordal/target/body-fitness-respawn-build-20260927/runtime-respawn-evidence/scenario-gates-4171-1790496039423805087/)では、誤flag併用、親数・policy・動的更新などの不適合とinstrumentを、WAV生成・device起動より前に拒否した。局所focused試験7件は、通常Rhaiに伴う初期actionの受理と後続変更の拒否、親pool所属、独立Tone積分に加え、4件のfault注入（実更新後energyと二度目の機会、支持・epoch・space・habituation・sourceの不一致、複数死亡・親不足・5 source目、実子のID・member・body・Recipe・候補欠落）を含み、[ログ](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-respawn/target/respawn-validation/focused-lib.log)に成功を記録した。既存の通常cleanupに対する親RNG reportの3 policy回帰も[単独試験1件](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-respawn/target/respawn-validation/parent-rng-regression.log)が成功した。先行v17cのintegration試験2件は[focusedログ](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-respawn/target/respawn-validation/focused-render.log)で成功し、v17fの全suite内でも対象integration試験2件は成功した。全suite全体も1322成功・0失敗・48 ignore、exit0（2026-09-27 17:05:17 JST）で終了した。format、標準Clippy、全target checkは各終了コード0を記録した。

v17fの全suite結果と固定物は[統合検証](body-fitness-offline-respawn-validation-20260927.md)へ保存した。登録した全境界の独立検証完了とは扱わない。上記の局所fault注入と通常renderの実経路は別の証拠である。worker欠測や自己PCM不一致など、ここで挙げた全境界を通常renderで発火させたという意味でもない。peak抽出の独立再実装とruntimeへの未注入faultは残る。性能・実device・長期生態・同ID再利用・出生policyの一般化も未測定。旧版の計時結果を第十七版の実時間合格へ転用しない。
