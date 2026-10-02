# F3e5: exact 候補密度の有界再利用 取得前登録

日付: 2026-09-27。状態: 取得前登録。F3e phase4 の source・raw・結果 capsule は変更しない。今回の cache は `cfg(test)` の明示配送とlive-paced試験内の身体密度に限る。通常runtimeへの採用を意味しない。

## 再利用できる量と失効境界

`representative_subjective_intensity` は実 Tone の `Recipe`（BodySnapshot、候補Hz、hold、ADSR、modulator、smoothing、fs）、source id から決まるModal位相 seed、観測 hop 数、`AnalysisStream` のNSGT kernel／RT smoothing／Log2Space／spectral front-end設定から密度を作る。`Identity::new` は Recipeとsource id/body generationをhashするが、source generation/birth、解析 epoch、観測 hop 数、AnalysisStream設定は含まない。従ってそれらを別のcache境界に置く。

sourceごとのcacheは `SourceIdentity`（id/generation/birth）、body generation、候補Hzを除いたRecipe、解析 epoch、観測 hop 数、**同一の不変 `Arc<AnalysisStream>` テンプレート**をcontextとする。テンプレートのstrong Arcを保持し、pointer再利用を防ぐ。テンプレートのpointer identityが違えば、同じ設定でも全失効する。この安全側missでNSGT kernel、空間、RT smoothing、front-endの全設定差を包含する。候補entryはexact pitch bitと、`recipe.freq_hz` をその候補Hzに置換して作った `Identity` を照合する。RNG/targetはlive Voiceから現時点で捕捉し、cache keyに過去の値を使わない。環境C、F2 score、habituation、受理済みF3b batchは保存しない。いずれも判断時のfresh値で作る。route/current pitch/target/RNG/habituationのVoice失効判定は弱めない。

既定 `Log2Space::new(55, 8000, 96)` は690 bin。phase4 の132候補だけで密度payloadが `132 × 690 × 4 = 364,320 byte` になるため、256 KiB では全候補が残らない。保持はsourceごと最大256 entryかつcache entry本体／scanの確保量1 MiB。強参照する共有AnalysisStreamテンプレート本体はこのentry予算と別に一つだけ保持する。entry本体のVec容量と密度Vec容量をbyte計上し、超過前に最終使用の古いentryを除去する。1 scanでもbyte上限に収まらなければ保存せず、次回再計算する。cacheはrunning jobと共にworker threadへ移し、完了時に戻す。一sourceにつき同時runningは1、Readyも最大1を保つ。候補全集合は毎Request現行RNG/target/landscapeからexact bitで列挙し、hitは密度Vecを複製、missのみ実 Tone の72 hopを計算する。環境score表は毎判断で再計算する。

## 機能検査

同じRequestをcacheあり／なしで仕上げ、密度全binのf32 bit、候補Identity、score全件、実Voiceのtarget／salience／adaptation／終了RNG／commit後基音を一致させる。二回目Requestでは共通候補にhit、新しい非格子候補はmissとし、hit/miss/evictionとsource別使用byteを記録する。Recipeの単独変更、body generation、source generation/birth、epoch、観測hop数、異なる空間またはfront-endテンプレートではhit 0を要求する。単なるqueue boundedや仮想Readyを性能成功としない。

## release live-paced 取得

phase4 と同じ固定Sine 440 Hzのsource 1／4、48 kHz・512 sample/hop、各938 hopをrelease test binaryで各一回。全hop実dt、身体・PCM一致、F3b非blocking受信、測定中密度thread joinなし、最終join後回収を維持する。先に1／4sourceのbuildと機能検査を終え、親が他のcargo/renderを停止した専有枠を指定してから取得する。元phase4のrawと並べ、cache hit/miss/eviction、実q再計算候補数、thread完了時刻/所要時間、窓内/窓外完了、source別Ready機会と**実数値消費**、延期gate、拒否理由、保持数/byte、F3b失効、deadline missとstart jitterを分離記録する。成功条件は各source実数値消費がphase4の5回を超え、全実Voice判断が有効な表を使用し、deadline miss 0、F3b失効0、保持上限遵守。満たさない結果も保持し、設定変更やbest run選択をしない。応答頻度を別に報告し、通常runtime・身体変化一般へ外挿しない。
