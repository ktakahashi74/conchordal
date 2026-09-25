# A1：I11-1／I11-2の関係の可聴性・実装前登録草案

状態：草案（未レビュー・未凍結）。実装・取得なし

2026-09-24の既存記録とWAVの対応を整理した、試聴手順の実装前登録案である。I11の機構実装前の登録を遡って作るものではない。音声再生、新規render、回答収集、提示鍵の生成・封印は行っていない。作者は既存60本による限定評価と4 session各15試行の交差手順を選び、続く「推奨で、進められるところまで進めて」により、T1の受け渡しとT2の生成音内のまとまりを検査する音楽的仮説として採用した。これは仮説の意味の選択であり、正確な提示文言・数値・提示順の凍結、登録全体の受入、可聴性の成立を意味しない。4／6以上等の数値案は未採用・未凍結である。

## 1. 目的・正本・版

対象はI11-1のT1 footprint（`body` 対 `proxy`）とI11-2のT2到来項（有効対無効）である。「違う音だった」「好みだった」だけでは通過とせず、意図した時間関係を聞き分けられるかを問う。作者採用A2、性能R2、実機受入A3は別に残す。I12bのA1は取得後の別登録とし、本書では§8の受け取り欄だけを置く。

正本は[milestones §1.3](milestones.md#13-機構選択規則による本体範囲の縮小2026-09-20同日のレビューを反映)と§3のA1行、および[設計メモ §10.3 Gate 3](../../design-notes/dcc-neurocognitive-hierarchy.md#103-four-independent-gates)である。本体A1は作者一人の試聴だけで完了可能であり、他者試聴は任意の参考である。複数聴取者、48人以上、募集・検出力・参加者clusterを前提としたO05〜O08を本体へ戻さない。ただし三記述、foil、不明回答、共同成功条件、因果介入と接続除去の区別は残す。

作者の確定判断(b)は[設計台帳](../../design-notes/technote-ledger.ja.md#body-beat-time-constants)に従う。今回のI11-1は合成器由来の代表footprintの版であり、学習したfootprintではない。現在のA4はこの版で閉じる。学習版はA4の後の次の単位とし、置き換え後にI11-1のA1と[R2のfootprint関連負荷](r2-preflight.md#41-測定対象版に固定するfootprintの出所)を再実施する。

[I4の分離](i4-recall-separation.md#r2-handoff-order)がまだでも既存版の試聴案は登録できるが、第二段階が生成に影響した場合、旧WAVの回答を新版へ自動継承しない。source／binary／入力／設定の一致または登録した回帰によって、証拠が適用できる範囲を別途確認する。

## 2. 条件だけで決める素材選定案

選択差、波形差の大きさ、好み、試聴結果による選別はしない。元の登録集合から、次の条件を満たすすべての組を採る案である。ファイルの欠損やSHA不一致は同じ層の別ファイルへの差し替え理由にせず、準備未完了として記録する。

- T1：同一record修正版の登録12条件を全件採る。sine-hold／sine-flow／harmonic-flow／modal-flow × 4／16／64 Voice、`render`、`rep=0` の `body`／`proxy` 24本。技術的な同一入力比較は[Mac取得記録](i11-onset-comparison.md#stage1-mac-validation-20260923)と[限定技術完了](i11-onset-comparison.md#stage1-technical-completion)による。同じ代表recordの識別・内容、D、16 bin、外部入力が一致するのは最初の分岐までであり、その後の全音響履歴まで同じとは主張しない。
- T1の主たる作用層は、元の登録で必須だったharmonic／modalのflow各3条件、計6対とする。sine-flowの3対は参加決定がある無差対照、sine-holdの3対は参加決定がない対照とし、三層を別集計する。これは既存の登録上の役割に従う層別であり、聞こえた条件だけを主判定へ入れる操作ではない。
- T1の補助比較：同じ12条件の修正版 `none` 12本と基準版 `none` 12本も全件採る。前者は `body` に対する広い生成作用の除去比較、後者との対はWAV不変の対照である。基準 `none` と `body` の異版比較を単独の因果介入にはしない。
- T2：I11-2の96本から、登録phaseが `offline_stage2_pulse` の新6条件だけを全件採る。sine／harmonic／modal-pulse-flow × 4／16 Voice、`stage2_candidate`、`rep=0` の `body-on`／`body-off` 12本である。旧条件や別binaryの同名configを混ぜない。4／16はpulse 1声を含む総数であり、対象Voiceは3／15声である。
- T2は全6対を主分母に残す。[§5.5](i11-2-results.md#55同一状態介入の作用)で無分岐だったsine-pulse-flow-4とharmonic-pulse-flow-4も除外しない。身体別・Voice数別も報告する。4対だけを選んだ成功率は主判定に使わない。

合計はT1 48本＋T2 12本の**60本**である。全体を先頭から終端まで使う案とし、分岐時刻に合わせた切り出し、無音の削除、個別の音量正規化、pulseの追加・混音をしない。登録は6秒本体＋0.5秒終端であり、実ファイルのsample数は準備時に検査して保存する。再生gainと出力機器は全条件で共通にする。今回の確認はSHAと取得記録の対応までで、WAVの聴取・decodeによる長さやclip検査は行っていない。

### 2.1 T2で聞こえる範囲

[新6条件の登録](i11-2-arrival-comparison.md)はpulseをhabitat専用、対象Voiceをpresentation専用とする。封印された第2段sourceの `src/runtime/mod.rs:3015–3016` はpresentationをWAVへ送っている。したがってこの12本に外部pulseそのものは含まれない。作者に2 Hzの基準音を別途聞かせると、新しい提示条件になり盲検にも影響するため、本案では行わない。

課題は、生成音の中で複数の音の入りがまとまるか、ずれた並びを作るか、その形が繰り返されるかに限る。外部pulseに近い位置と、全体が同じ量だけずれた位置は、不可聴の基準を補わずには区別できない。したがって前案の「繰り返しの目安に近い／ずれた位置」は、外部基準との関係と生成音内の関係を混同するため使わない。「外部pulseとの位相関係を直接聞いた」とも解釈しない。

A1-3で選択した限定評価の範囲を維持し、作者が選択した生成音内のまとまりを§4の仮説とする。到来費用が変わることから、複数の音が可聴なまとまりを作ることまでは導けない。外部pulseへの接近そのものは本素材だけでは未判定であり、必要なら後続の提示条件と素材を別登録する。今回はpulseの追加・混音・新規renderをしない。

### 2.2 素材検査の受入契約案

- 提示準備時に、§10の60項目をpath・case・variant・取得版で一意に照合し、原WAVと提示用複写のSHAを登録値へ照合する。同じSHAの別variantも別試行として保持し、内容の重複を理由に減らさない。欠損・重複割当・SHA不一致・登録にない素材は準備未完了とし、差し替えない。
- 取得版のwriter契約であるmono・PCM16と登録48 kHzを確認し、WAV全データがdecodeでき、headerのframe数と実際のframe数が一致して正であることを検査する。`frames, sample_rate, duration=frames/sample_rate, peak_abs, full_scale_count`を原WAV SHAとともに準備記録へ保存する。`duration`は秒、`peak_abs`はPCM整数の絶対値最大、`full_scale_count`は絶対値が32767以上のsample数である。登録の6秒＋0.5秒から実WAVが厳密に312,000 sampleだとは仮定せず、取得版の終了処理と記録へ照合する。理由不明の長さの差は準備未完了として調べ、切り詰め・水増しで合わせない。
- peakと整数PCMの最大振幅値の件数は診断であり、それだけでclipの発生や可聴性を判定しない。閾値や「差のある音」の選択条件を後付けしない。再生系の動作確認と共通gain設定には本60本を使わず、対象素材を聴く前に出力機器・gainを固定する。今回これらの実ファイル検査・再生確認は行っていない。

## 3. 作者一人向けの条件秘匿と提示案

元のGate 3は、一人の聴取者へ対の片方だけを割り当てる。作者一人ではそれを厳密に保つと素材差と介入差を同じ対で比較できない。本案は**別日に対のもう片方を提示する交差手順**へ適合させる。元の複数聴取者設計と同等とは扱わず、先行聴取の記憶が残る限界と、作者が技術的結果を既に知る限界を記録する。他者の参加は必要条件にしない。

- 4 session各15試行とする。各sessionにT1の12条件を1本ずつ置き、同じcaseのbody／proxy／修正版none／基準noneを4 sessionへ一つずつ割り当てる。各sessionは四variant各3本とし、T1作用層6条件では各variantが1本または2本になるよう均衡化する。noneだけのsessionを作って条件を明かさない。body対修正版noneは補助比較として別集計し、主介入の不成功を救済しない。
- T2は各sessionに各身体1本、計3本を置く。各身体の4／16 Voice×on／offの4本を4 sessionへ分散する。各sessionのon数は1本または2本とする。同じcaseの対は別sessionになる。全体の条件順・素材順を制約内で乱数化する。
- 主介入の各対について、body／onが先になるものとproxy／offが先になるものを、T1作用層6対・T2全6対でそれぞれ3対ずつにする。各対の先に聞く側だけの集計も残す。ただしT1ではnoneを既に聞いている場合もあり、完全な初見ではない。既聴のvariant数と順序も記録する。
- session間は48時間以上を第一案とする。各session内の順序と三記述の表示順を独立に乱数化し、同じ条件の対を連続提示しない。各試行は最大2回の全長再生、回答後の戻り再生なし、答え合わせなしとする。48時間は記憶消去を保証する値ではない。
- 作者の反復聴取を独立聴取者数へ換算しない。選択済みの交差手順を、実施結果に応じて片方だけの集計へ置き換えない。

上記の本数・variant均衡・先行順の制約は、抽象caseによる構成例で同時成立を確認した。実素材への提示割当や鍵は作っていない。制約を満たすことは、記憶効果の除去や可聴性の成立を意味しない。

後続の準備器は、作者に乱数seedや対応表を表示せず、非公開の高entropy seedから提示ID・順序・割当・選択肢順を作る設計とする。入出力の形は以下の案であり、今回は準備器や鍵を作らない。

- 入力 `registration`：本書の確定版SHA、素材一覧（原path・WAV SHA・case・条件・binary版・役割）、正確な三記述と固定ID、分母・判定規則、session制約、再生設定、素材検査記録、§9の固定欄。登録の実byte列をSHA-256の対象とし、そのSHAは登録本体に自己埋め込みせず公開commitmentに置く。
- 秘匿 `key.json`：schema版、seed、ランダムsalt、各sessionの提示ID順、提示ID→原path／WAV SHA／case／条件／役割／表示位置→記述ID、生成器版。数値・文字encoding・改行・key順を固定した実byte列をSHA-256の対象にする。同じ登録・生成器・seedから同じ割当を再現し、制約を満たす最初の割当を使う。制約に合う割当を見て引き直さず、回答取得後に再生成しない。
- 公開 `commitment.json`：鍵のSHA-256、登録版SHA、生成器SHA、試行数、作成時刻。鍵とseedは全回答確定後まで表示しない。鍵のSHAは事後差し替え検出であって暗号化ではないため、鍵を開かない運用と提示側からの分離が必要である。
- 提示 `trials`：無意味なランダムID、再生用不透明ID、三記述と「わからない」だけを表示する。原path、variant、case、原SHA、ファイルサイズ、取得順、期待される記述、対の相手を表示しない。§10の素材一覧や元WAVを照合しながら回答しない。原ファイル名をランダム名へ変えるだけで元SHAを表示する設計も採らない。
- 準備検査：原素材60項目と提示IDが一対一であり、重複・欠損がなく、sessionごとの15本・T1のvariant数・作用層配分・T2の身体数・on数・各対の分離・主介入の先行順3対ずつを満たすことを検査する。各提示の三記述は対応T行の固定IDの順列であり、不明回答は一つだけである。検査結果の公開部分は成否と件数だけとし、対応表は出さない。
- 出力 `responses`：§6の回答。全sessionの回答を確定し、そのSHAと確定時刻を保存してから鍵を開く。未完了で中止する場合も欠損を含む記録を先に確定し、開封後の回答追加はしない。開封後は回答を上書きせず、修正は別の注記にする。鍵のSHA不一致、途中の対応漏洩、回答を見た再乱数化は無効な実施として保持する。

既存WAVやこの草案の条件表を作者が知っていても、提示IDと条件の対応は事前に見せない。ただし音色や人数から素材を推測できることまでは防げない。条件を推測した試行・既聴の記憶がある試行も消さず、§6に記録する。

## 4. 三記述・foil・方向の案

提示画面では「意図」「foil」の名称を出さない。以下の固定IDは解析用であり、表示位置は毎試行変える。各行とも四つ目は同じ「わからない」である。共通の教示案は「聞こえた音全体の関係に最も近い記述を一つ選ぶ。どれにも当てはまらない、または判断できない場合は『わからない』を選ぶ」とする。T1・T2の仮説の意味は作者選択済みであり、以下はその提示文言案である。文言の凍結と試聴はまだ行っていない。

- T1-I（意図）：「音の余韻が重なりながら、次の音へ動きが受け渡される。」
- T1-F1：「音の立ち上がりがそろい、まとまって動き出す。」
- T1-F2：「長い音は重なっているが、次の音へ動きがつながらない。」
- T2-I（意図）：「複数の音がそろって入り、そのまとまりが繰り返される。」
- T2-F1：「複数の音がずれて入り、その並びが繰り返される。」
- T2-F2：「複数の音が次々に入るが、まとまりや並びは繰り返されない。」

T1はbodyでT1-Iの選択がproxyより増える、T2はonでT2-Iがoffより増える方向を選択済みの仮説とする。これは技術実装の選択差から必然的に導かれる可聴効果ではない。とくに重なり費用の変更が「受け渡し」、到来項が「音の入りのまとまりの反復」として聞こえるかを検査する。T2の旧草案「異なる位置を保つ」は機構の方向と合わず、次の修正案「目安に近い」も不可聴の外部基準を含み得るため、選択した仮説では生成音内の関係だけを問う。試聴も回答取得もしておらず、聞こえた結果に合わせた正答変更ではない。逆方向やfoil優勢を得てから意図を付け替えない。静止したhold等に当てはまらなければ、不明回答のまま残す。

T1は元のGate 3の記述を仮説として残す。T2の選択は元の一般例とは異なるため、このI11-2用の関係として試聴前に明示登録する。Gate 3も一般例と異なる作者の比較には事前登録を求めている。T2-F2は「反復しない関係が聞こえた」という回答であり、「混み合っていて関係を判別できない」という不明回答とは区別する。音の入りや音群を分けて聞けない場合にF2を強制しない。文言だけの理解確認でこの区別を確認するが、対象音による練習や正答の教示は行わない。選択済みの音楽的意図は再確認の待ち項目にしない。対象音を聴いてから文言を変更した場合、既聴という事実は消せないため、その回答を新文言の未見検証へ算入せず、同じ60本の再試聴を独立な新検証とも呼ばない。作者好みの形容詞、機構名、body／proxyやon／offを示す語を加えない。

### 4.1 機構との対応を静的に確認した範囲

- T1：I11-1 §2・§4.4の重なり費用は `coupling * 6 * s * Ov / norm` であり、body／proxyは自声powerの形を置き換える。候補間の順位は変わり得るが、bodyなら全候補で重なりが減る、または「受け渡し」が増えるという単調な関係はない。したがってT1-Iは機構の同義語ではなく、作者が検査する音楽的仮説である。bodyの一般的優越性も主張しない。
- T2：I11-1 §4.5は「同期を促す方向だけ」を明記する。I11-2 §3の追加費用は `coupling * arrival_weight * (1 - P)` であり、係数が正で他の費用が同じなら、候補窓内の到来確率Pが高いほど安くなる。到来から離れる位置の維持を、この項自身の意図とは扱えない。総費用にはbaseも残るため、全onsetの同期・完全な同時発音・可聴な差は保証されない。
- T2では外部pulseがWAVにないため、選択した仮説で問うのは生成音の反復のまとまりである。潜在する外部pulseの位相への接近は、精度を問わず聴取だけでは判定しない。全Voiceが同じ到来項を参照しても、機会時刻・base費用・skipが異なるため、音の入りのまとまりは別に検査する仮説である。

原登録・数式・提示busの静的確認と作者の仮説選択は済んだが、登録全体の受入完了や可聴性の確認は意味しない。提示文言と数値等の凍結を残し、三草案の状態は未レビュー・未凍結のままとする。

## 5. 分母と共同成功条件の案

作者一人の、固定素材・固定版に限定した工学的判定とする。参加者clusterの95%区間、母集団の有意差、80%検出力をこの件数から主張しない。同じ音源の再生や反復回答は分母を増やさない。T1とT2は独立に判定し、一方の成功で他方を通さない。

主分母はT1作用層6対（body 6、proxy 6）、T2全6対（on 6、off 6）である。各armの割当全試行を分母Nとし、`p_I = n_I/N`、`p_F = (n_F1+n_F2)/(2N)` とする。「わからない」はNに含め、foil票にも意図票にも数えない。未回答は不明回答へ自動置換せず実施未完了とする。各armの四回答の整数件数と率を併記し、1/3・1/4の一様な偶然率は仮定しない。

関係同定の支持には、次の二条件を**同時に**要求する案とする。

1. body／onで `p_I - p_F > 0`。
2. `p_I(body/on) - p_I(proxy/off) >= 0.15`。

0.15は元のGate 3から採った実用差の候補値であり、作者一人に校正済みの閾値ではない。N=6では条件1は `2*n_I > n_F1+n_F2`、条件2は意図票の純増が1票以上という整数条件になる。例えばbody／onが意図1・不明5、proxy／offが不明6でも両条件を満たす。また、意図2・foil1が3・foil2が0・不明1でも条件1を満たし、個別foilより意図が多いことまでは要求しない。これは元の平均foil式の誤記ではなく、信頼区間を用いない作者一人への適合で生じる判定の弱さである。

追加の候補として、body／onで意図4／6以上も要求する案を推奨する。全割当試行の過半数で同定されることを要求し、上の少数票だけの通過を防ぐためである。校正済みの可聴性閾値でも元Gate 3の必須条件でもなく、A1-2で選び凍結するまでは未採用の案である。元の二条件だけを選ぶ場合は、この弱さを受け入れる判断を記録し、「登録した二条件を満たす」という限定結果とする。別の最低件数や不明上限を採る場合も回答取得前に固定する。

四択件数、身体別・Voice数別、全12条件を含むT1の補助集計に加え、対ごとの四択回答の遷移と、意図選択が増えた対・減った対・変わらない対の件数を示す。一票の純増を全6条件に一貫した効果へ読み替えない。各対の先に聞く側だけの結果（各arm 3件）も別掲し、その介入差が逆方向なら、全session合算で選んだ主条件を満たしても「先行提示の比較と方向不一致・保留」とする案である。この感度比較も素材差・既聴の影響を受けるため、順序効果の因果的同定とは呼ばない。任意の追加sessionを繰り返して解消しない。

無差対照の再現性も別に確認する。登録済みのcase・variant対で結合し、同じSHAの全組合せを新しい対へ増やさない。T1のsine-flow／sine-hold計6対では同一の四択回答が5対以上、T2のWAV同一2対では2対とも一致、修正版none対基準noneの12対では10対以上の一致を第一案とする。不明同士も一致に含み、記述同士の一致数と不明同士の一致数を分けて記録する。各層の分母は6・2・12のままとし、欠損対を消して一致率を上げず、欠損があれば未完了とする。複数の対照に同じ試行が現れても独立な回答数へ合算しない。これらは回答の安定性の候補閾値であり、達成しても関係効果の証拠にはならない。未達は保留とし、最終値は試聴前に選ぶ。

最終登録で選んだ主条件・最低件数・対照条件を満たしても、接続除去の対応が未確認なら、結果欄は「登録した関係同定条件を満たす／接続除去は未判定」とする。A1全体の合格へ繰り上げない。逆方向・差不足・採用した最低件数の不足は「登録した関係効果は支持されず」、欠損・秘匿破れ・先行提示との方向不一致・対照条件不足は「保留」と区別する。秘匿破れは個別trialを消して続行せず、実施全体の盲検判定を無効として保持する。A2の好みで判定を覆さない。

### 5.1 因果介入と接続除去の対応

- T1のbody対proxyは、同じrecord・D・外部入力でfootprintのpowerを置き換える介入である。proxyも重なり項を持つため、接続を外した条件ではない。
- T1のbody対修正版noneは、onset比較作用全体を外した補助比較である。重なり項だけを切り、他の候補生成・分割・打切りを保った除去とは同一でない。T1作用層で `p_I(body)-p_I(none)>=0.15` を補助基準案として報告できるが、単独で接続除去の成立を判定しない。より狭い除去素材が必要と判断した場合は別登録の残作業とし、今回はrenderしない。
- T2のbody-on対body-offは、生成側の到来項を有効／無効にする介入であり、その作用を外した比較でもある。ただしoffではCDF自体が存在しなくてよいという登録である。同じ非到来入力prefixの比較と、on側の同一CDFを使った独立介入検査が技術的な橋渡しとなる。上流状態を保った接続だけの切断と同一とは呼ばない。
- I12bのforward／backwardの接続除去はこのT2介入から別に扱う。§8を受領してから別のA1登録を作り、本書のon／offだけでI12b二経路のA1を完了させない。

A1-3の選択に従い、既存素材では限定した関係同定と広い除去の評価を先行する。厳密な接続除去をどの後続登録で補うかは残る。作者一人という人数の縮小を理由に、接続除去の必要性まで暗黙に削らない。

## 6. 試聴シートと判定記録の形

盲検中の1試行1行は次の形を案とする。条件欄は回答時には存在させない。

- sessionヘッダ：`session_id, registration_sha256, key_sha256, started_at, ended_at, pauses, output_device, gain, previous_session_ended_at, i12b_measurement_free_window`。`pauses`に中断・再開時刻と理由を残す。
- 回答行：`trial_id, displayed_descriptions, selected_display_position, cannot_tell, play_count, full_play_count, answered_at, heard_relation_note, remembered_material, suspected_condition, interruption`。位置は1〜3、不明は位置空欄・`cannot_tell=true`、未回答は位置空欄・`cannot_tell=false`とする。位置と不明の同時指定は不正な記録であり、補完しない。確定回答はtrialごとに一件、全長再生1回以上を要する。未知のID・重複回答・表示記述の不一致は検査エラーとして原記録を保持する。
- 任意の短い自由記述は聞こえた時間関係について記す。好き嫌いは主回答に混ぜず、必要なら鍵の開封後に別のA2メモへ書く。再生失敗、無回答、途中中断は独立欄に残す。
- 回答確定記録：`responses_sha256, locked_at, completed_trial_count, missing_trial_ids`。
- 開封後の解析行：上記に `case, arm, role, wav_sha256, source_version, selected_description_id` を結合する。T1三層、T2全6対、none対照を分け、N・四択件数・二つの主差・採用した最低件数・対ごとの遷移・対照一致数・先行提示の感度比較・秘匿逸脱・接続除去の適否を保存する。一致の判定は表示位置でなく固定記述IDで行う。

自由記述を見て正答を変更したり、「わからない」を正答に読み替えたりしない。中断・欠損には次の契約案を用いる。

- `play_count`は音声送出開始イベントの回数であり、途中停止も一回に数える。`full_play_count <= play_count <= 2`とする。再開時も同じtrial ID・順序・表示記述・残り再生回数を保ち、回答済みtrialへ戻らない。
- 未回答trialは、残り枠があれば全長を先頭から再生できる。全長を一度も再生できず枠が尽きたtrialは欠損のまま残し、不明票や別素材に置換しない。全長再生済みなら再生を追加せず回答を確定できる。次のtrialへ進む場合も欠損記録を残す。
- session中断後の再開は同じsessionの残り順序で行い、終了時刻は最後の再開後の終了とする。次sessionまでの48時間案はそこから測る。機器・gain・登録SHA・鍵SHAが変わった場合や秘匿が破れた場合は、同じ実施を継続しない。
- 全60件の有効回答がそろわない場合、実施全体は「未完了」とし、観測済み件数・欠損・逸脱を保存する。T1の48件・T2の12件は別に確認し、必要な回答と対照が欠けた側の合否は出さない。完答だけで分母を作り直さず、一方の欠損を他方の不成功へ読み替えない。中止して鍵を開いた後は回答を追加せず、後日の再試聴を初回の続きや未見の新検証にしない。

## 7. 実施前の運用条件

**試聴はI12bの性能取得（instrument 144本）と時間を重ねない。** WAV再生も同じMacのCPUを使う。I12b担当側の測定終了または測定を行わない時間帯を確認してから開始し、確認時刻・実施区間を記録する。電源がBatteryという過去の状態や、見えている出力がないことから空き時間を推定しない。進行中プロセスを停止して空き時間を作らない。

実施は登録・秘匿手順のレビューと固定の後の別作業である。素材のコピーやhash検査等の準備も性能取得と重ならない時間へまとめる案とする。新規render、ファイル加工、他者の募集は本案の実施に含めない。本人の回答収集と、研究拡張の人的収集手続きを混同しない。

## 8. I12b A1の受け取り欄

すべて未記入である。取得本数・技術判定・資源判定を先取りしない。

- source／binary／登録・検査器manifestのSHA：未受領。
- 取得完了数、失敗・逸脱・除外理由：未受領。
- full／forward-cut／backward-cutの対応するcase・seed・入力・WAV path／SHA：未受領。
- 二経路の切断定義、保持した上流状態、最初の分岐、同一入力の範囲：未受領。
- 試聴で聞けるbus、関係記述・foil・不明回答・接続除去の共同判定案：未登録。
- I11との版対応、性能取得終了または測定のない時間帯：未確認。

## 9. 判断点と終了点

2026-09-24、作者の「推奨」により前回1〜4と次の三件の推奨方針を選択した。前回5〜7はその選択に従って扱い、前回8〜9の詳細は後続のレビュー・登録に残す。作者判断(b)も維持する。方針選択は三草案全体のレビュー完了や登録凍結を意味しない。

- **A1-1（新10、方針選択済み）：素材集合と作者一人の提示法。** 60本全件・4 session各15試行の交差案を作者限定の記述的比較として進める。片方だけを提示する代案は選ばなかった。対内比較と各対の先行提示だけの感度確認を残す。記憶・順序の限界も残り、48時間間隔等の運用詳細は登録前に確認する。
- **A1-2（新11、仮説の意味も選択済み）：関係記述と成功規則。** 作者の「推奨で、進められるところまで進めて」により、T1の「受け渡し」と、T2の「生成音内で音の入りがまとまって反復する」を各介入の音楽的仮説として採用した。本人意図の判断待ちは解消した。外部pulseとの位相関係は既存WAVでは未判定のままである。数値は§5の二条件に意図4／6以上を加える案を第一候補とするが、意味の選択から数値の承認を推定しない。最低件数・0.15・対照一致数・先行提示の扱いは未採用・未凍結であり、登録凍結時の採否に残す。
- **A1-3（新12、方針選択済み）：既存WAVの判定範囲。** 関係同定と広い除去の限定評価を先行し、厳密な接続除去は残作業とする。試聴自体を必要素材の別登録後まで待つ代案は選ばなかった。T1の広いnone除去、T2の聞こえない外部pulse、I12b未受領の限界を合格へ読み替えない。接続除去の免除ではない。
- 登録凍結時の必要欄：60項目の版・SHAと素材検査記録、§4の三記述・固定ID・教示・方向、§5の分母・式・最低件数の採否・対照閾値・先行提示と未判定の扱い、4 sessionの制約・48時間案・再生上限、機器・gain・休憩・§6の欠損と再開規則、準備器／提示器／集計器の版・鍵保存先・回答確定方法、性能取得のない実施区間を固定する。未記入の欄を既定値で埋めて凍結済みにしない。鍵と実際の順序は固定した登録と生成器からその後に作る。

本書の終了点は、版と素材の追跡、秘匿手順・回答形式・判定規則・不足する比較がレビュー可能になった状態である。試聴実施・登録凍結・A1合格・A2採用・A4完了を意味しない。

## 10. 素材の実パス・SHA-256・条件

以下は準備者用の対応表であり、盲検中の提示画面へ出さない。パスはリポジトリルートからの相対パスである。2026-09-24に `shasum -a 256` で60本を読み取り照合した。T2の12本は保存ledgerのSHAと一致し、T1のnone両版24本はbit-identity記録のSHAと一致した。T1のbody／proxy 24本は今回の実ファイルSHAを列挙したもので、過去の個別WAV SHAの独立再監査とは区別する。

取得台帳はT1修正版 `target/i11-same-record-mac-20260923/runs.json`（36件）、T1基準 `target/i11-local-baseline-20260923/none12/runs.json`（12件）、T2 `target/i11-stage2-20260924/offline-ledger-96-reconstructed.json`（96件、うち登録sequence 85〜96を選定）である。後者は取得後に復元された96本分のledgerであり、取得前snapshotと呼ばない。

T1修正版renderのSHAは `29f15c2af1246601164d1769790fe1860db341ae544671c4ac533862b667a688`、基準renderは `1800f993407e756e64c266abc556ca37d469aefeef711d5d1e5b4669045ecb31`。T2候補renderは `1d1d810f234d13d0947cba507715d2a94aaae11627fef82a457dc48ba2ce4e1e` である。これらは取得台帳・結果文書の識別値で、今回binaryを実行・再buildしていない。

基準commit表記のずれは、追加の静的確認で出所を特定した。`bit-identity.json` の `baseline_commit.given = 32a6389` は、検査器 `i11-inputs/bit_identity.py:158` が `plan.get("baseline_commit")` を転記した値であり、比較先binaryから検出したcommitではない。`--baseline-dir` は別引数である。現在の登録planにも同じ旧IDが残る。タグ `i11-stage1-baseline` は `103a32fe5c5fc4fdf0f4d65fb25a1bb0c7c47303` を指し、タグ注記は旧IDが履歴の付け替え前の同じ登録に対応し、src・Cargo.toml・Cargo.lockは同一、差はbuildが読まないPDFだけだったと記録している。

今回、基準取得が使った `target/i11-local-baseline-20260923/none12/render` と保存buildの `conchordal-render` を読み取りhash照合し、両方が上記 `1800f993…` と一致することを確認した。取得の `binaries.json`、基準manifest（SHA-256 `9d3bb481a828ebf305a70d87d0592a4a81ea957a140ce22fba212ccbcb9e9327`）、タグの指すcommitも整合する。旧IDは現在のローカルgitでは解決できないため、旧新版sourceの同一性そのものはタグ注記と保存検証記録による来歴であり、今回の独立な再比較ではない。A1の版同定は実際の基準commit・binary SHAとWAV SHAで行い、旧planのIDを実行版と誤記しない。元の取得記録・plan・manifestは変更していない。

### 10.1 I11-1：同一record版12条件と基準対照

全件 `render, rep=0`。各条件のbody／proxyが主介入対、body／修正版noneが補助的な広い除去対、修正版none／基準noneが不変対照である。sine-flowとsine-holdのbody／proxyは同一SHAであるが、両ラベルを残す。

- **harmonic-flow-16**：T1作用層。
  - body：`target/i11-same-record-mac-20260923/harmonic-flow-16-body-render-0.wav` — SHA-256 `7f7c1bc4c14679e5975bc08bcd7c8fbbd6f1faad0e227676a7720cea278e32ba`。
  - proxy：`target/i11-same-record-mac-20260923/harmonic-flow-16-proxy-render-0.wav` — SHA-256 `e9b70bdf53fbe3fa772177b57915192672b727de84553ed00a4b4ba4cb1782ce`。
  - 修正版none：`target/i11-same-record-mac-20260923/harmonic-flow-16-none-render-0.wav` — SHA-256 `c7d6e7fab04653ed8b8d32e9eb758d391d2699fddfba2cd7b16d46cb6175f2ac`。
  - 基準none：`target/i11-local-baseline-20260923/none12/harmonic-flow-16-none-render-0.wav` — SHA-256 `c7d6e7fab04653ed8b8d32e9eb758d391d2699fddfba2cd7b16d46cb6175f2ac`。

- **harmonic-flow-4**：T1作用層。
  - body：`target/i11-same-record-mac-20260923/harmonic-flow-4-body-render-0.wav` — SHA-256 `e8d1189adc1413eaa23045665ed8ed076c2dd4c3eef38695f32c1bfc48cfe142`。
  - proxy：`target/i11-same-record-mac-20260923/harmonic-flow-4-proxy-render-0.wav` — SHA-256 `d2e3c445cdf8070ee6ce9bd9fd199f6b94e5133af92ef655a407f8fd1852e4c1`。
  - 修正版none：`target/i11-same-record-mac-20260923/harmonic-flow-4-none-render-0.wav` — SHA-256 `0082a88ba6963bde23ccf47ae01492f76531798d3b918d75ec07ddb344677ee3`。
  - 基準none：`target/i11-local-baseline-20260923/none12/harmonic-flow-4-none-render-0.wav` — SHA-256 `0082a88ba6963bde23ccf47ae01492f76531798d3b918d75ec07ddb344677ee3`。

- **harmonic-flow-64**：T1作用層。
  - body：`target/i11-same-record-mac-20260923/harmonic-flow-64-body-render-0.wav` — SHA-256 `c258d2ff0d23a10e0a9049e5ed763befb456c1e3e77829711f350ee18d610b9f`。
  - proxy：`target/i11-same-record-mac-20260923/harmonic-flow-64-proxy-render-0.wav` — SHA-256 `1266e7c0f6ddaeb80a3c02d3a43ef3a0587029886f03ff88156417bf31ca1888`。
  - 修正版none：`target/i11-same-record-mac-20260923/harmonic-flow-64-none-render-0.wav` — SHA-256 `879d2da16d6e708a0d406f40e9a6cb00092c763a3f98c7db5f1c148de2c11447`。
  - 基準none：`target/i11-local-baseline-20260923/none12/harmonic-flow-64-none-render-0.wav` — SHA-256 `879d2da16d6e708a0d406f40e9a6cb00092c763a3f98c7db5f1c148de2c11447`。

- **modal-flow-16**：T1作用層。
  - body：`target/i11-same-record-mac-20260923/modal-flow-16-body-render-0.wav` — SHA-256 `2d06192df08403cb8a48070c688a8aea4b58281edb581a586305f8b5733d932d`。
  - proxy：`target/i11-same-record-mac-20260923/modal-flow-16-proxy-render-0.wav` — SHA-256 `4386e9544ece5a7d890ecd609ee62d0e023ab2399fcff846e7ed1c7cbf635978`。
  - 修正版none：`target/i11-same-record-mac-20260923/modal-flow-16-none-render-0.wav` — SHA-256 `374206d1c94c53714797a6c891b41ddef9883b653dbaae44f027a19cd05203d1`。
  - 基準none：`target/i11-local-baseline-20260923/none12/modal-flow-16-none-render-0.wav` — SHA-256 `374206d1c94c53714797a6c891b41ddef9883b653dbaae44f027a19cd05203d1`。

- **modal-flow-4**：T1作用層。
  - body：`target/i11-same-record-mac-20260923/modal-flow-4-body-render-0.wav` — SHA-256 `080fad954eef83044eaf5effcae737e4d5634a0c8057c8cf1df2a5c836a852cc`。
  - proxy：`target/i11-same-record-mac-20260923/modal-flow-4-proxy-render-0.wav` — SHA-256 `566a5711bc9c54ab72489cb887742017f7b2f934159d5cc69a3c10388effb098`。
  - 修正版none：`target/i11-same-record-mac-20260923/modal-flow-4-none-render-0.wav` — SHA-256 `889f348e6bd9dfa28b35edcd705e6e0e824f21c962c3e94528f047f86c9c6a29`。
  - 基準none：`target/i11-local-baseline-20260923/none12/modal-flow-4-none-render-0.wav` — SHA-256 `889f348e6bd9dfa28b35edcd705e6e0e824f21c962c3e94528f047f86c9c6a29`。

- **modal-flow-64**：T1作用層。
  - body：`target/i11-same-record-mac-20260923/modal-flow-64-body-render-0.wav` — SHA-256 `ad03a6d68bce7cfdfbcd5507986cac6fbc35c470e5ee5bb0520eb67cd2b58c8d`。
  - proxy：`target/i11-same-record-mac-20260923/modal-flow-64-proxy-render-0.wav` — SHA-256 `1712093b1fde56e17a87727e4c87d4786bd258900c20364f1242976801b63780`。
  - 修正版none：`target/i11-same-record-mac-20260923/modal-flow-64-none-render-0.wav` — SHA-256 `f7d09b0a7b3ea8fa2442b7045b156d19337c56ef8c92d4c4a94905aeb63b2a56`。
  - 基準none：`target/i11-local-baseline-20260923/none12/modal-flow-64-none-render-0.wav` — SHA-256 `f7d09b0a7b3ea8fa2442b7045b156d19337c56ef8c92d4c4a94905aeb63b2a56`。

- **sine-flow-16**：参加決定あり・無差対照。
  - body：`target/i11-same-record-mac-20260923/sine-flow-16-body-render-0.wav` — SHA-256 `1695e9e91a2ad860978526ed6bf1256a517e93727a4e4615334078b027b835eb`。
  - proxy：`target/i11-same-record-mac-20260923/sine-flow-16-proxy-render-0.wav` — SHA-256 `1695e9e91a2ad860978526ed6bf1256a517e93727a4e4615334078b027b835eb`。
  - 修正版none：`target/i11-same-record-mac-20260923/sine-flow-16-none-render-0.wav` — SHA-256 `b9db26b5d3b7a210da76ab0c801021ae89e3994bb4050cad5e77a16a4903e3f8`。
  - 基準none：`target/i11-local-baseline-20260923/none12/sine-flow-16-none-render-0.wav` — SHA-256 `b9db26b5d3b7a210da76ab0c801021ae89e3994bb4050cad5e77a16a4903e3f8`。

- **sine-flow-4**：参加決定あり・無差対照。
  - body：`target/i11-same-record-mac-20260923/sine-flow-4-body-render-0.wav` — SHA-256 `3dc4ec57c5c584049a0179ed1472b85b610952e3c606d24059b36fa0b95e81d5`。
  - proxy：`target/i11-same-record-mac-20260923/sine-flow-4-proxy-render-0.wav` — SHA-256 `3dc4ec57c5c584049a0179ed1472b85b610952e3c606d24059b36fa0b95e81d5`。
  - 修正版none：`target/i11-same-record-mac-20260923/sine-flow-4-none-render-0.wav` — SHA-256 `47046e854f2885a667e727391ef2914131c72c089fb63649e9ea1eaff6049b7a`。
  - 基準none：`target/i11-local-baseline-20260923/none12/sine-flow-4-none-render-0.wav` — SHA-256 `47046e854f2885a667e727391ef2914131c72c089fb63649e9ea1eaff6049b7a`。

- **sine-flow-64**：参加決定あり・無差対照。
  - body：`target/i11-same-record-mac-20260923/sine-flow-64-body-render-0.wav` — SHA-256 `7f82c35b920438520b0f62bfa0dab7a76cdf5c592105c9572184683666df6ca5`。
  - proxy：`target/i11-same-record-mac-20260923/sine-flow-64-proxy-render-0.wav` — SHA-256 `7f82c35b920438520b0f62bfa0dab7a76cdf5c592105c9572184683666df6ca5`。
  - 修正版none：`target/i11-same-record-mac-20260923/sine-flow-64-none-render-0.wav` — SHA-256 `918f8d86db6c3dec0ac7ebeb93cfd6fb5acb50c62ad416ca539fc2480d5f0fe1`。
  - 基準none：`target/i11-local-baseline-20260923/none12/sine-flow-64-none-render-0.wav` — SHA-256 `918f8d86db6c3dec0ac7ebeb93cfd6fb5acb50c62ad416ca539fc2480d5f0fe1`。

- **sine-hold-16**：参加決定なし対照。
  - body：`target/i11-same-record-mac-20260923/sine-hold-16-body-render-0.wav` — SHA-256 `6352a29dee55f5ec4392d7752263872ada648c0be9e9f725e6eadc03d446c4fe`。
  - proxy：`target/i11-same-record-mac-20260923/sine-hold-16-proxy-render-0.wav` — SHA-256 `6352a29dee55f5ec4392d7752263872ada648c0be9e9f725e6eadc03d446c4fe`。
  - 修正版none：`target/i11-same-record-mac-20260923/sine-hold-16-none-render-0.wav` — SHA-256 `6352a29dee55f5ec4392d7752263872ada648c0be9e9f725e6eadc03d446c4fe`。
  - 基準none：`target/i11-local-baseline-20260923/none12/sine-hold-16-none-render-0.wav` — SHA-256 `6352a29dee55f5ec4392d7752263872ada648c0be9e9f725e6eadc03d446c4fe`。

- **sine-hold-4**：参加決定なし対照。
  - body：`target/i11-same-record-mac-20260923/sine-hold-4-body-render-0.wav` — SHA-256 `66176d060373f59094a0d450c48e9690ca72eced449bbce154d496a7411df82b`。
  - proxy：`target/i11-same-record-mac-20260923/sine-hold-4-proxy-render-0.wav` — SHA-256 `66176d060373f59094a0d450c48e9690ca72eced449bbce154d496a7411df82b`。
  - 修正版none：`target/i11-same-record-mac-20260923/sine-hold-4-none-render-0.wav` — SHA-256 `66176d060373f59094a0d450c48e9690ca72eced449bbce154d496a7411df82b`。
  - 基準none：`target/i11-local-baseline-20260923/none12/sine-hold-4-none-render-0.wav` — SHA-256 `66176d060373f59094a0d450c48e9690ca72eced449bbce154d496a7411df82b`。

- **sine-hold-64**：参加決定なし対照。
  - body：`target/i11-same-record-mac-20260923/sine-hold-64-body-render-0.wav` — SHA-256 `b351b91aaba0c16740c7536528ab652fac1b0fbaf8882e84d064b257cfa16032`。
  - proxy：`target/i11-same-record-mac-20260923/sine-hold-64-proxy-render-0.wav` — SHA-256 `b351b91aaba0c16740c7536528ab652fac1b0fbaf8882e84d064b257cfa16032`。
  - 修正版none：`target/i11-same-record-mac-20260923/sine-hold-64-none-render-0.wav` — SHA-256 `b351b91aaba0c16740c7536528ab652fac1b0fbaf8882e84d064b257cfa16032`。
  - 基準none：`target/i11-local-baseline-20260923/none12/sine-hold-64-none-render-0.wav` — SHA-256 `b351b91aaba0c16740c7536528ab652fac1b0fbaf8882e84d064b257cfa16032`。

### 10.2 I11-2：新pulse全6条件

全件 `offline_stage2_pulse, stage2_candidate, render, rep=0`。body footprint、同じ登録Hazard、`arrival_weight=1.0`、`arrival=true`（body-on）対 `false`（body-off）である。無分岐2対も主分母に残す。

- **harmonic-pulse-flow-16**：登録した到来介入対。
  - body-on：`target/i11-stage2-20260924/acquisition/092-harmonic-pulse-flow-16-body-on-render-0.wav` — SHA-256 `35333af4bdeccb97dfa224357d332888d1275548c3e6e9f05eb855070df3a9ac`。
  - body-off：`target/i11-stage2-20260924/acquisition/091-harmonic-pulse-flow-16-body-off-render-0.wav` — SHA-256 `bedb8f05f2d62d8323f9c2b95d7487ec2e17829d645d1c8a3316acf69e2217a0`。

- **harmonic-pulse-flow-4**：登録上の無分岐・WAV同一対。
  - body-on：`target/i11-stage2-20260924/acquisition/090-harmonic-pulse-flow-4-body-on-render-0.wav` — SHA-256 `3853145f413c3c72c15e297609cac9636a090887a5dcdba00af09451e4782808`。
  - body-off：`target/i11-stage2-20260924/acquisition/089-harmonic-pulse-flow-4-body-off-render-0.wav` — SHA-256 `3853145f413c3c72c15e297609cac9636a090887a5dcdba00af09451e4782808`。

- **modal-pulse-flow-16**：登録した到来介入対。
  - body-on：`target/i11-stage2-20260924/acquisition/096-modal-pulse-flow-16-body-on-render-0.wav` — SHA-256 `a37763f90dd4e54897d32f6ed7329f96681a1bf5c1d4133f86fc0a955a3d2218`。
  - body-off：`target/i11-stage2-20260924/acquisition/095-modal-pulse-flow-16-body-off-render-0.wav` — SHA-256 `4a12a7fbc1ce6326d4b3e23958de3f17e8eca480a35a20b8614d573362c39098`。

- **modal-pulse-flow-4**：登録した到来介入対。
  - body-on：`target/i11-stage2-20260924/acquisition/094-modal-pulse-flow-4-body-on-render-0.wav` — SHA-256 `945b153ce573c6609eb88b56d452b671d6f18b07bb949e1e887106a2661bb754`。
  - body-off：`target/i11-stage2-20260924/acquisition/093-modal-pulse-flow-4-body-off-render-0.wav` — SHA-256 `b3d55ac267365d036b9d91f7eb44e348d89043317005e80a105b84e73243c55e`。

- **sine-pulse-flow-16**：登録した到来介入対。
  - body-on：`target/i11-stage2-20260924/acquisition/088-sine-pulse-flow-16-body-on-render-0.wav` — SHA-256 `50db34ca603d7ffbb6247b9654a93c603f151250e0b0212c16c4420e99639af2`。
  - body-off：`target/i11-stage2-20260924/acquisition/087-sine-pulse-flow-16-body-off-render-0.wav` — SHA-256 `8e4d17eabac037980268654e6a843c94e5ef23f3614bf8ea3d3f98554d79f1a0`。

- **sine-pulse-flow-4**：登録上の無分岐・WAV同一対。
  - body-on：`target/i11-stage2-20260924/acquisition/086-sine-pulse-flow-4-body-on-render-0.wav` — SHA-256 `aa84af434607e51d970340d24e5db484713db47e0df9b3818dc14739f2e3a90c`。
  - body-off：`target/i11-stage2-20260924/acquisition/085-sine-pulse-flow-4-body-off-render-0.wav` — SHA-256 `aa84af434607e51d970340d24e5db484713db47e0df9b3818dc14739f2e3a90c`。
