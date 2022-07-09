#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
library(ggplot2)
library(tidyverse)
library(magrittr)
library(dplyr)
library(DT)
library(ggrepel)
library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)
library(stringr)
library(data.table)
library(ggpubr)
library(DataExplorer)
library(readr)
library(lubridate)
library(FSA)
library(agricolae)
library(hrbrthemes)
library(heatmaply)
library(factoextra)
library(caret)
library(kableExtra)
library(nloptr)

library(shiny)
library(shinydashboard)
options(repr.plot.width=8, repr.plot.height=3,shiny.reactlog = TRUE)

rsconnect::setAccountInfo(name='financialbigdataanalysisproposal', token='1EAD8DFC027FA4145BA21E5181486163', secret='GlEZp76irJImkIUNLDUdIaNKprIA3/2B8+jvs/Pj')

library(rsconnect)

ui <- dashboardPage(
  skin = 'yellow',
  dashboardHeader(title = "顧客區隔"),
  dashboardSidebar(
    
    sidebarSearchForm(textId = "searchText", buttonId = "searchButton",
                      label = "Search"),
    
    sidebarMenu(
      menuItem("摘要", tabName = "summary", icon = icon("newspaper"), selected = TRUE),
      menuItem("研究主題", tabName = "theme", icon = icon("thumbtack")),
      menuItem("研究動機", tabName = "motivation", icon = icon("lightbulb")),
      menuItem("研究方法", tabName = "method", icon = icon("chart-line"),
               menuSubItem("數據清理", tabName = "DATA_CLEANING"),
               menuSubItem("顧客交易資料分析", tabName = "TRANSACTION_ANALYSIS"),
               menuSubItem("顧客取消交易資料分析", tabName = "CANCELLATION_ANALYSIS"),
               menuSubItem("顧客交易金額及種類分析", tabName = "STOCK_CODE_AND_TOTAL_PRICE"),
               menuSubItem("顧客價值分析: RFM模型", tabName = "RFM_MODEL"),
               menuSubItem("顧客交易時間分析", tabName = "TIME_ANALYSIS"),
               menuSubItem("顧客交易地區分析", tabName = "COUNTRY_ANALYSIS"),
               menuSubItem("顧客區隔", tabName = "CUSTOMER_SEGMENTATION"),
               menuSubItem("交易資料文字雲", tabName = "WORD_CLOUD")),
      menuItem("結論", tabName = "conclusion", icon = icon("clipboard")),
      menuItem("附錄", tabName = "others", icon = icon("paperclip")),
      menuItem("四個問題", tabName = "question", icon = icon("question"))
    )
  ),
  dashboardBody(
    tabItems(
      # First tab content
      tabItem(tabName = "summary",
              h2("摘要"),
              h4("以顧客分群為題，將原始資料進行清洗，觀察原始資料的交易量、交易時間及交易金額，分析不同要素對於行銷策略的影響。利用RFM模型進行顧客價值分析，並以k-means演算法進行顧客分群，找出目標客群以進行更好的行銷。")
      ),
      
      # Second tab content
      tabItem(tabName = "theme",
              h2("電商市場顧客區隔"),
              tabsetPanel(
                tabPanel(h4("顧客區隔"),
                         h3("一、什麼是顧客區隔？"),
                         h4("顧客區隔也叫做Customer Segmentation，傳統的的顧客區隔主要以年齡層、收入、居住區域來劃分，而破壞式創新大師克里斯汀生主張從顧客心中的待完成工作(Jobs to be done)，也就是說顧客不再是購買商品，而是把顧客當成老闆，他們出費用來「僱用」商品或者服務來完成他們目前需要完成的工作，顧慮需要的不是商品本身，而是這個商品能給我們帶來的附加價值與功能。"),
                         br(),
                         h3("二、為什麼要有顧客區隔？"),
                         h4("現在的市場更新迭代的很快，比起想方設法的去招攬潛在的新顧客，不如穩固當下的老顧客。畢竟當新顧客成了老顧客，而你提供的產品或者服務不能滿足他們新的需求，那麼顧客將不會為你停留。"),
                         br(),
                         h4("對目標市場分析，互聯網打通了顧客與企業之間的溝通渠道，顧客可以透明的和企業溝通，而這時如何針對客戶營銷就成了商家必須要面對的問題，顧客區隔的理論也就隨之而產生。有了區隔才可以放大產品的價值。"),
                         br(),
                         h3("三、顧客區隔的五種類型都是哪些？"),
                         h4("顧客區隔的類型主要為年紀、性別、學歷、特色、行為這五種因素。要知道不同的顧客群體，對於同一種營銷有著不同的反應，我們在實際中可以根據這五種因素來分類客戶群，進而制定不同的行銷策略精準模型。"),
                         br(),
                         h4("就像共享房屋Airbnb軟體一樣，它並不是只是提供簡單的租房房源，他的定位十分明確，可以說它是通過軟體來為顧客提供了一項服務或者說提供了一種方案。根據不同的消費群體，它會推出不同的房源，就像有的人喜歡體驗當地生活，有的人更在意性價比。"),
                         br(),
                         h3("四、怎樣做好顧客區隔"),
                         h4("在做顧客區隔之前應該先收集大量的目標消費族群的數據資訊，對目標市場分析才能做好顧客區隔。"),
                ),
                tabPanel(h4("RFM 模型"),
                         h4("RFM 模型是由喬治卡利南(George Cullinan)於1961年所提出，在眾多的客戶關係管理(CRM)模型中，RFM 是最常被使用的，因為它對顧客的描述較為完整，且不只是著重在顧客貢獻的金額。接下來，會詳細解釋 RFM 作為數據分析的三個要素:"),
                         br(),
                         h4(tags$b("Recency(上次消費的日期)")),
                         h4("指的是最近一次顧客消費的時間。近期有消費過的顧客，會是最有可能再次回購的人，因此行銷專員可以針對這群人發放折價優惠或是商品訊息。"),
                         br(),
                         h4(tags$b("Frequency(消費頻率)")),
                         h4("最近顧客常到實體店面或是線上消費嗎？消費頻率越高的人就越有可能是品牌的忠誠顧客，也因此可以透過行銷策略，持續提高這群人的購買意願。"),
                         br(),
                         h4(tags$b("Monetary(消費金額)")),
                         h4("消費金額的多寡過去常被視為較重要的要素，但是結合 Recency、Frequency 就會是一個完整的顧客消費輪廓。未來若是要採用分眾行銷，RFM 模型就會是個參考依據。"),
                         
                ),
              )  
              
      ),
      tabItem(tabName = "motivation",
              h2("研究動機"),
              h4("現代電商競爭激烈，五花八門的資訊對消費者來說不易進行消化，不同型態的消費者對於電商業者而言也不容易找到對的客戶，因此我們希望能夠以online retail II 這個資料及為例，藉由課堂中學習到的數據分析及資料視覺化的方法進行此店商平台的顧客價值分析，並進行顧客分群，以達到更好的行銷效果。")
      ),
      tabItem(tabName = "method",
              h2("method tab content"),
      ),
      tabItem(tabName = "DATA_CLEANING",
              h2("數據清理"),
              h4("這次使用的數據是由英國電商公司所提供的顧客交易數據，年份為2009至2011，數據中包含na values，因此在開始使用數據前我們先看一下缺少的數據為那些項目以及其比例。"),
              dataTableOutput("missing_table"),
              dataTableOutput("missing_table_1"),
              h4("在2009-2010年間缺少的數據比例大約為20%"),
              img(src="missing_1.png"),
              h4("在2010-2011年間缺少的數據比例大約為25%"),
              img(src="missing_2.png"),
              h4("因此，我們將這些缺少的數據進行清除並畫圖重新檢查目前使用的資料是否仍有缺少的數據。"),
              img(src="missing_clean.png"),
              h2("數據資訊"),
              h4("可以透過以下此表簡單的瞭解這次使用的數據所包含的架構。其中Invoice代表的是發票號碼，StockCode為商品代碼, Description為顧客該筆交易所包含的所有產品名稱，用逗號分開， Quantity為購買數量，InvoiceDate為發票日期，Price, Customer.ID, Country則為價格、顧客代碼及顧客所在國家。"),
              dataTableOutput("data_info"),
      ),
      tabItem(tabName = "TRANSACTION_ANALYSIS",
              h2("顧客及產品數量統整"),
              h4("在這筆資料中總共包含5942位不同的消費者，他們購買了4646種商品，並總共進行了約45000筆交易。"),
              dataTableOutput("product_info"),
              h2("國家平均交易數量"),
              h4("令人意外的是，此電商平台的為英國所開發，然而前十大平均交易量之國家卻沒有包含英國。"),
              dataTableOutput("product_num"),
              h4("以前10大交易量的國家進行畫圖，黃色虛線代表所有有在平台進行交易的國家他們的平均交易量，紅色虛線則代表交易量前十大的國家之交易量平均值。"),
              img(src="country_mean.png"),
              h2("顧客平均交易數量視覺化"),
              h4("以前10大交易量的之顧客代碼進行畫圖，黃色虛線代表所有有在平台進行交易的顧客他們的平均交易量，紅色虛線則代表交易量前十大的顧客之交易量平均值。"),
              img(src="customer_mean.png"),
              h2("主要顧客來源"),
              h4("由上圖可知，前十大交易量顧客隻交易量平均遠高於所有顧客隻交易量平均，那這些顧客是從哪裡來的呢?以顧客代碼作為篩選條件進行資料整理，下表顯示大量交易之顧客來源。"),
              dataTableOutput("top_customer"),
              h4("如同我們預期，大部分的交易量還是來自於英國，因此可以推知這家店商應該是以國內市場為主，為此在後續的k-means分群應以客戶消費行為分群，國家因素的影響性應該不大。"),
              h2("只被買一次的產品"),
              h4("在研究交易量的同時我們也好奇，有些顧客只在平台上進行一次交易，是因為他們買的不是消耗品嗎?購物體驗不好?抑或是商品品質不好?下表列出只進行過一次交易的顧客，他們買的商品包含哪些。"),
              dataTableOutput("product_purchase_once"),
              h4("由上表可知，這些商品大部分為裝飾性商品、玩具、包裝紙等等，其中包裝紙佔消耗性商品的大宗。")
      ),
      tabItem(tabName = "CANCELLATION_ANALYSIS",
              h2("交易取消之資料"),
              h4("在資料處理的過程中我們發現了幾件有趣的事:
                 1.發票號碼的首位若包含C則代表這筆交易曾經被取消過。2.有些消費者只有到此店商消費一次而且只買過一樣東西，相反的也有消費者一次購買大量商品。
                 因此我們想分析曾經被取消過的交易是否有甚麼含意。"),
              h2("顧客取消交易數量"),
              h4("下表為顧客取消交易之數量，其中取消交易的訂單佔總交易訂單約17%，是一個不小的比例，因此這個問題需要被重視。"),
              dataTableOutput("cancel_order"),
              h2("取消的訂單都有被買回?"),
              h4("在處理資料的過程中我們發現當有一筆資料被一位顧客取消時，有另一筆相同交易商品資料被同樣的顧客代碼購回，所以我們推測顧客取消訂單可能是因為外部因素，而不是因為商品本身，因此我們假設所有取消的訂單都會被購回。"),
              verbatimTextOutput("cancel_check"),
              h4("然而就資料分析的結果而言，並不是所有取消的訂單都有被購回，因此我們需要再分析顧客取消訂單之原因並進行改善。"),
              h2("訂單取消數之國家排名"),
              h4("從下表可知，訂單取消數最多的國家為德國，因此此電商平台可以針對德國消費者隻交易模式進行更進一步的分析，以留住顧客。"),
              dataTableOutput("cancel_country"),
              h2("清除取消"),
              h4("由於取消的訂單不應該被作為後續顧客分類之資料，因此我們將被取消的訂單進行清除。"),
              verbatimTextOutput("doubtful_cleaning")
              
      ),
      tabItem(tabName = "STOCK_CODE_AND_TOTAL_PRICE",
              #h2("Stock code分析"),
              #dataTableOutput("stock_code"),
              h2("交易金額分析"),
              h4("利用ggplot畫出此平台消費者的交易金額分布圖，x軸為交易金額區間，y軸為該交易金額區間之交易數量。由此可知，主要的交易金額落在15-20美元，電商平台可以以此做為目標顧客的設定條件之一。"),
              #dataTableOutput("check_consume_range"),
              img(src="price_cut.png"),
      ),
      tabItem(tabName = "RFM_MODEL",
              h2("RFM模型分析"),
              h4("在進行顧客區隔前，我們先利用RFM模型進行顧客價值分析，所謂的RFM是指Recency、Frequency、Monetary，因此我們將此電商平台的消費數據進行顧客價值分析。"),
              img(src="plot_3h.png"),
              h4("由於資料分布有些歪斜，因此我們利用Log函數標準化它。"),
              img(src="monetary_log.png"),
              h2("Cluster Dendrogram"),
              h4("我們利用階層式分群法(hierarchical clustering)的架構，將資料層層反覆地進行分裂或聚合，以產生最後的樹狀結構，來分析不同客戶之間的交易關係，由下圖可知，此平台的消費者彼此之間存在著交易上的關聯性。"),
              img(src="cluster_dendrogram.png"),
              img(src="group_1.png"),
              img(src="rfm_analysis.png"),
              h4(tags$b("你80%的收入來自於你20%的客戶。")),
              h4("RFM 分析幫助公司合理地預測哪些客戶可能再次購買他們的產品，有多少收入來自新（相對於重複）客戶，以及如何將偶爾購買的顧客變成習慣性購買者"),
              h4("分析主要是根據客戶最近一次的交易時間、來店消費的頻率以及消費金額去對客戶進行排名和分組。這邊舉一些例子示範可能顧客種類會表現的出行為："),
              h4("-鯨魚. (R, F, M) = (高, 高, 高) 鯨魚客群，近期客戶，常常消費，且花費高。相應的廣告活動可能就會以提醒他們自已經有一段時間沒有購物，再透過特價或優享活動為他們提供恢復購買的動力。
"),
              h4("-新客戶。(R, F, M) = (高、低、-)，近期才來消費的，歸類為新客戶。有針對性的對其宣傳可能將新客戶變成常客。Group(1, 4, 9)
"),
              h4("-流失的客戶。 （低、-、高）的客戶曾經是有價值的客戶，但後來流失掉了。這時對曾購買物品有針對性的行銷可能衝新激勵客戶回流。Group(5, 7)
"),
              h4("企業可以使用 RFM 分析來識別具有相似行為的客戶群，並針對特定客戶類型製定有針對性的行銷活動。它使企業能夠使用電子郵件或實體郵件的行銷來定位大量特定類型的客戶更有可能響應的消息。"),
              
              h4("根據RFM三種因子為每個客戶評分，以提供賣家進行分析。"),
              #h2("recoding"),
              textOutput("recoding")
      ),
      tabItem(tabName = "TIME_ANALYSIS",
              h2("營收分析(日)"),
              h4(""),
              img(src="revenue_by_date.png"),
              h2("營收分析(週)"),
              h4(""),
              img(src="revenue_day_week.png"),
              h2("交易及客戶數量、營收箱型圖(週)"),
              h4(""),
              img(src="revenue_box_plot.png"),
              h4(""),
              img(src="revenue_summary.png"),
              h2("無母數K-W檢定"),
              h4("因為p-value小於5%顯著水準，我們有足夠資料說明顧客的平均消費行為(週)不一致。"),
              #verbatimTextOutput("k_test"),
              h2("Tukey確實差異檢定"),
              h4("可以看出相較其他星期，週四顧客的消費行為有顯著的差異。"),
              #verbatimTextOutput("tukey"),
              h2("總收入比例圖(週)"),
              h4(""),
              img(src="revenue_pie.png"),
              h2("總交易量比例圖(週)"),
              h4(""),
              img(src="transaction_pie.png"),
              h2("總顧客數量比例圖(週)"),
              h4(""),
              img(src="customer_pie.png"),
              h2("小結"),
              h4("一個禮拜內不同日子的交易數量和產生的收入在統計上存在顯著差異。從圖片可以分析知道週六擁有最少的交易數量，週四則最高。由於`average order value`相對沒什麼波動，我們合理推測交易數量的不同與當天創造的收入相關。"),
              h2("時間序列分析(小時)"),
              h4("鑑於周末的交易數量較少，而星期四的交易數量較高，我們可以針對電傷的廣告支出分配提供建議。"),
              img(src="hour.png"),
              h4("可以觀察出，早上到下午我們有較多的交易和收益，然而在趨近晚上交易量大幅減少。如果這是每日趨勢，或能提供減少夜間推播產品廣告的建言。")
      ),
      tabItem(tabName = "COUNTRY_ANALYSIS",
              h2("國家交易頻率資料"),
              h4("根據下表資料，我們可以看出主要交易的消費者國家為英國"),
              #dataTableOutput("country_data"),
              h4("我們利用點狀圖和條狀圖將不同國家的交易頻率視覺化"),
              img(src="country_point.png"),
              img(src="country_bar.png"),
              h2("國家交易紀錄資訊比較"),
              #dataTableOutput("country_view"),
              h2("前五國分析"),
              #dataTableOutput("top5country_summary"),
              h2("前五國資料視覺化"),
              img(src="country_value.png"),
              h2("平均交易所得來源"),
              img(src="country_avg_trans.png"),
              h2("交易量排名前五國資料總結"),
              img(src="country_summary.png"),
              h2("平均交易所得來源箱型圖"),
              img(src="country_order_box.png"),
              h2("平均交易量來源箱型圖"),
              img(src="country_trans_box.png"),
              h2("平均顧客來源資料總結"),
              img(src="year_summary.png"),
      ),
      tabItem(tabName = "CUSTOMER_SEGMENTATION",
              h2("顧客花費金額分析"),
              img(src="revenue_per_customer.png"),
              h2("顧客交易量分析"),
              img(src="num_of_trans.png"),
              h2("例子：顧客No. 12380購買物品、數量及花費"),
              #dataTableOutput("customer_eg"),
              h2("總資料分析"),
              #verbatimTextOutput("customer_analysis"),
              h2("顧客分群熱圖"),
              h4("透過這張heatmap，可以看到我們將電商的總收入與預期的交易數量分為一群，`MeanRev` 和 `MedianRev` 分為另一群。最後再透過分析回傳的數據，期望可以發現行為方式相似的客群。"),
              img(src="heatmap.png"),
              h2("實驗1：最優集群"),
              h4("將總資料全部放入模型裡進行的分群結果。綠色的群心幾乎落在藍色區域的邊緣，一些與紅色集群歐式距離較近的點卻被歸類於綠色，集群區分效果並不好。"),
              img(src="optimal_cluster.png"),
              h2("實驗1：K-means"),
              img(src="cluster_plot.png"),
              h2("實驗2：K-means(RFM)"),
              h4("這次改將RFM分析結果餵入模型計算，並改成使用6個cluster"),
              img(src="rfm_optimal_cluster.png"),
              h2("實驗2：K-means with RFM"),
              h4("分類準確度有明顯提升，然而仍有一些距離較遠的點需要被歸類。"),
              img(src="rfm_cluster.png"),
              h2("實驗3：K-means with whole data"),
              h4("彙總實驗1及實驗2使用的資料進行計算。"),
              img(src="whole_cluster.png"),
      ),
      tabItem(tabName = "WORD_CLOUD",
              h2("自然語言處理"),
              h4("在分析顧客交易產品相關性的同時做一個有趣的應用，我們將產品敘述進行文字切割，並列出常出現在產品名稱中之單字列表。"),
              #dataTableOutput("nlp_head"),
              h2("互動式文字雲"),
              h4("將單字列表及單字出現之頻率做成文字雲，可以拖動滑桿產生不同的文字雲。"),
              # sliderInput("freq",
              #             "Minimum Frequency:",
              #             min = 1,  max = 50, value = 15),
              # sliderInput("max",
              #             "Maximum Number of Words:",
              #             min = 1,  max = 300,  value = 100),
              # plotOutput("plot"),
              h2("2009-2012顧客最常消費的產品特徵或產品"),
              h4("最後將常出現的單字以條狀圖進行資料視覺化。"),
              img(src="word_frequency.png"),
      ),
      tabItem(tabName = "conclusion",
              h2("結論"),
              h4("由種種分析我們可以知道了解一些主要顧客的消費習慣，如消費時間點、消費地區及平均消費金額，利用這些資訊進行網頁行銷活動規劃。")
      ),
      tabItem(tabName = "others",
              h2("附錄"),
              h4(tags$a(href="https://www.ecfit-saas.com/2020/12/17/rfm/", "1.	RFM 是什麼？分眾行銷把顧客「標籤化」案例分析！")),
              h4(tags$a(href="https://tw.strikingly.com/content/blog/customer-segmentation/", "2.	顧客區隔")),
              h4(tags$a(href="https://zh.wikipedia.org/zh-tw/K-%E5%B9%B3%E5%9D%87%E7%AE%97%E6%B3%95", "3.	K-means 演算法")),
              h4("作者:107070013盧庭誼 108048110郭玫均")
      ),
      tabItem(tabName = "question",
              h2("四個問題"),
              h3(tags$b("(1) Where you get your project ideas?")),
              h4("平時會利用蝦皮或pchome等電商軟體購買商品，好奇電商公司的推薦系統是怎麼進行推薦，以及針對不同的顧客會不會以有購物時間點的偏好等等，因此希望從一個電商軟體的角度，進行顧客數據的分析，研究看看不同消費者的消費習慣以進行顧客分群。"),
              br(),
              h3(tags$b("(2) What R techniques have been used?")),
              dataTableOutput("pack_used_output"),
              br(),
              h3(tags$b("(3) How many new packages and functions are used?")),
              dataTableOutput("new_pack_output"),
              br(),
              h3(tags$b("(4) What is the most difficult part of your analysis?")),
              h4("最困難的部分是希望座購買產品的關聯性分析，由於發票明細的產品名稱可能包含數字、顏色、公司名稱等等，在產品名稱的命名規則雜亂的情況下，不容易取出主要的產品名稱，因此比較難進行相關性的分析。除此之外，在使用shiny的部分，將網站上線至shiny官網有滿多環境問題，需要花費很多時間克服。"),
      )
    )
  )
)

server <- function(input, output) { 
  new_pack <- readxl::read_excel('new.xlsx')
  pack_used <- readxl::read_excel('packages used.xlsx')
  output$new_pack_output <-renderDataTable({
    datatable(new_pack)
  })
  output$pack_used_output <-renderDataTable({
    datatable(pack_used)
  })
  
  # ############################################################code_run_before#####################################################
  year2009_2010 <- read.csv('online 2009 (1).csv')
  year2009_2010 <- rbind(year2009_2010, read.csv('online 2009 (2).csv'))
  year2009_2010 <- rbind(year2009_2010, read.csv('online 2009 (3).csv'))
  year2009_2010 <- rbind(year2009_2010, read.csv('online 2009 (4).csv'))
  year2009_2010 <- rbind(year2009_2010, read.csv('online 2009 (5).csv'))
  year2009_2010 <- rbind(year2009_2010, read.csv('online 2009 (6).csv'))

  year2010_2011 <- read.csv('online 2010 (1).csv')
  year2010_2011 <- rbind(year2010_2011, read.csv('online 2010 (2).csv'))
  year2010_2011 <- rbind(year2010_2011, read.csv('online 2010 (3).csv'))
  year2010_2011 <- rbind(year2010_2011, read.csv('online 2010 (4).csv'))
  year2010_2011 <- rbind(year2010_2011, read.csv('online 2010 (5).csv'))
  year2010_2011 <- rbind(year2010_2011, read.csv('online 2010 (6).csv'))

  year2009_2010_show <-year2009_2010
  year2010_2011_show <-year2010_2011
  year2009_2010 <- as.data.frame(year2009_2010[complete.cases(year2009_2010),])
  year2010_2011 <- as.data.frame(year2010_2011[complete.cases(year2010_2011),])

  CustomerData <- rbind(year2009_2010, year2010_2011)

  # Info
  coltype <- t(lapply(CustomerData, class))
  null_values <- t(as.matrix(lapply(CustomerData, \(x){sum(is.na(x))})))
  null_values_p <- sapply(null_values, \(x){x/dim(CustomerData)[1]})

  Info_table <- rbind(coltype, null_values, null_values_p)
  row.names(Info_table) <- c('column type', 'null values (num)', 'null_values (%)')

  # Check for duplicate entries and delete them
  as_tibble(CustomerData) #824,364 × 8
  CustomerData <- CustomerData %>% distinct() #797885*8

  # num of users and products
  products <- CustomerData$StockCode %>% unique %>% length
  transactions <- CustomerData$Invoice %>% unique %>% length
  customers <- CustomerData$`Customer.ID` %>% unique %>% length
  Info_customer_product <- as.data.frame(cbind(products, transactions, customers))
  
  
  # there are 5942 different customers, they bought 4646 different products and that the total
  # number of transactions carried out is of the orer of approximately 45000
  
  
  # determining the num of product purchased in every transactions

  transactions <- CustomerData %>%
    group_by(Country,Invoice) %>%
    summarize(n=n()) %>%
    group_by(Country) %>%
    summarize(Mean = mean(n)) %>%
    arrange(desc(Mean))

  country_table_show <- transactions

  # Surprisingly, top 10 countries who made the most avg transactions do not include
  # UK!!!
  transactions <- CustomerData %>%
    group_by(`Customer.ID`,Invoice) %>%
    summarize(n=n()) %>%
    group_by(`Customer.ID`) %>%
    summarize(Mean = mean(n)) %>%
    arrange(desc(Mean))

  transactions$`Customer.ID` %>% as.character() -> transactions$`Customer.ID`

  # the avg transaction made by top 10 customers is way more than the average.
  # Where does these top 10 customers came from?
  CustomerData$`Customer.ID` %>% as.character() -> CustomerData$`Customer.ID`
  CustomerData_show <- CustomerData
  transaction_country <- function(CustomerData, num=10){
    CustomerData %>%
      group_by(`Customer.ID`, Country) %>%
      filter(`Customer.ID` %in% head(transactions, num)$`Customer.ID`)%>%
      transmute(Country = Country) %>%
      group_by(Country) %>%
      summarise(Customers=n_distinct(`Customer.ID`),
                proportion = Customers/num) %>%
      arrange(desc(proportion))
  }



  # As we can expect, most of the transactions are made from UK, but
  # among top 100 of customers, customers from UK still take up about 93% of the
  # transactions
  # 可以得知這間ecommerce的客群仍然以國內市場為主
  # 因此推測kmeans分群應以客戶消費行為分群，國家因素的影響性應該不大。

  #############################################
  #############################################

  ## products that only been purchased once
  ## pick Customers that shop only one time what did they buy?

  CustomerData %>%
    select(`Customer.ID`, Invoice) %>%
    group_by(`Customer.ID`) %>%
    summarise(transactions = n()) %>%
    filter(transactions == 1) %>%
    select(`Customer.ID`) |>
    sapply(FUN = unname) -> never_come_back

  CustomerData[never_come_back,] %>%
    group_by(Description) %>%
    summarise(Quantity = sum(Quantity),
              Price = mean(Price)) %>%
    arrange(desc(Quantity)) |>
    head(30) %>%
    knitr::kable()
  
  transactions <- CustomerData %>%
    group_by(`Customer.ID`,Invoice) %>%
    summarize(n=n())

  transactions$canceled_order <- sapply(transactions$Invoice, \(x) {sum(grepl('C', x))})
  # num of order canceled: 4372/23587 (17.60%)
  sum(transactions$canceled_order) / dim(transactions)[1]
  # noticed that the num of cancellation is quite large (~17.6% of the total num of
  # transactions)
  cancel_order_show <- transactions

  # we observe from the data that when an order is canceled, another
  # transactions with the same customerID and description then shown in the dataframe,
  # mostly identical except for the **Quantity** and **Invoice Date** variables
  CustomerData[178:196,]

  # Check if all the cancelled orders appears on the bottom of the data set.
  # by locating the negative quantity entries and check if there is
  # systematically an order indicating the same quantity with the same description
  data_check <- as.data.frame(CustomerData %>% subset(Quantity<0))
  data_check_use <- data_check
  for(row in row.names(data_check)){
    customerID <- data_check[row, ]$`Customer.ID`
    descrip <- data_check[row, ]$Description
    quantity <- -data_check[row, ]$Quantity
    size = CustomerData %>% filter(`Customer.ID` == customerID,
                                   Description == descrip,
                                   Quantity == quantity) %>% dim
    if (size[1]==0){
      cat(paste(data_check[row,], '\n'))
      cat("-----> Hypothesis not fulfilled.")
      break
    }
  }

  # We found that the initial hypothesis is not verified.
  # (ie. cancellations do not necessarily correspond to orders that would have
  # made afterwards.)

  data_check <- CustomerData
  data_check$Cancelled_Q <- 0 # cancelled Q check

  doubtful_entry <- read.csv("doubtful data.csv")
  entry_to_remove <- read.csv("entry to remove.csv")
  cancelled_list <- read.csv('index to add cancelled quantity.csv')
  doubtful_entry <- doubtful_entry$x
  entry_to_remove <- entry_to_remove$x
 

  # ################################ Some Invetigation on cancellations
  # ################################
  # ################################
  data_check[cancelled_list$Index, ] %>%
    group_by(Country) %>%
    summarise(cancelled_transactions = n_distinct(Invoice),
              cancelled_customers = n_distinct(`Customer.ID`)) %>%
    mutate(total_cancelled = sum(cancelled_transactions),
           cancelled_proportion = round(cancelled_transactions/total_cancelled,3),
           total_cancelled=NULL) %>%
    arrange(desc(cancelled_transactions)) %>% #|> head(8)
    filter(Country!="United Kingdom") %>%
    head ->cancel_country_show

  # Apart from the UK, Germany has the most cancellations country of order

  # to be continued.........

  ################################ Some Invetigation on cancellations
  ################################
  ################################

  # doubtful entry list correspond to the entries indicating a cancellation but
  # there is no command or order afterwards.
  # so I would delete all of the doubtful entries and entry_to_remove
  (doubtful_entry %>% length)/nrow(data_check)*100 # count for %0.23 of the datafarme entries
  (entry_to_remove %>% length)/nrow(data_check)*100 # count for 2.07% of the dataframe entries

  doubtful_entry = as.integer(doubtful_entry)
  entry_to_remove = as.integer(entry_to_remove)
  remove_data_vec <- c(doubtful_entry, entry_to_remove)

  add_cancelled_Q <- function(data, cancelled_list){
    data$Cancelled_Q <- 0
    data[cancelled_list$Index,]$Cancelled_Q = -1*cancelled_list$cancelled_Q
    return(data)
  }

  data_check = add_cancelled_Q(data_check, cancelled_list)

  sum(data_check$Cancelled_Q) # 420754

  dim(data_check) # 797885*10
  length(remove_data_vec)# 18390
  data_check %>% filter(Quantity<=0) %>% nrow # 18390
  data_check = data_check[-remove_data_vec,]
  dim(data_check) #779495*10

  CustomerData = add_cancelled_Q(CustomerData, cancelled_list)
  CustomerData = CustomerData[-remove_data_vec,]
  doubtful_cleaning_show <- CustomerData
  #CustomerData %>% glimpse #779495*9

  ############################*********************************########################
  # data_check %>% mutate(Total_price = Price*(Quantity-Cancelled_Q), .before=1) -> data_check
  # 
  # # check the range
  # data_check$Total_price %>% max # 38970
  # data_check$Total_price %>% min # 0
  # 
  # data_check %>%
  #   group_by(`Customer.ID`, Invoice, Price) %>%
  #   summarize(Total_price = (Quantity-Cancelled_Q)*Price,
  #             Date = mean(InvoiceDate)) -> new_data
  # 
  # # testing cutting numbers with 5-9 groups, we decided
  # # to cut 9 groups by numbers of observations to gove relatively same proportion to each group
  # cut_number(new_data$Total_price, 9) %>% unique %>% sort
  # 
  # data_check %>%
  #   group_by(Description, Price) %>%
  #   summarize(#Total_price = (Quantity-Cancelled_Q)*Price,
  #     num = n()) -> new_data
  # # implementing RFM
  # 
  # range(CustomerData$InvoiceDate)
  # 
  # data_check %>%
  #   group_by(`Customer.ID`) %>%
  #   summarise(recency = as.numeric(as.Date("2011-12-09")-as.Date(max(InvoiceDate))),
  #             freq = n_distinct(Invoice),
  #             monetary = sum(Total_price)/n_distinct(Invoice)) -> data_RFM
  # 
  # data_RFM %>% summary
  # 
  # data_RFM_scaled <- data_RFM
  # row.names(data_RFM_scaled) <- data_RFM_scaled$`Customer.ID`
  # 
  # 
  # data_RFM_scaled$`Customer.ID` <- NULL
  # data_RFM_scaled <- scale(data_RFM_scaled)
  # summary(data_RFM_scaled)
  # 
  # cluster <- hclust(dist(data_RFM_scaled), method='ward.D2')
  # 
  # #plot(cluster, hang=-1)
  # 
  # summary(cluster)
  # 
  # # cut a tree into groups of data
  # members = cutree(cluster, k=9)
  # 
  # table(members)
  # 
  # aggregate(data_RFM[,2:4], by=list(members), mean) -> RFM_m
  # 
  # ## recoding and character variables to factors
  # data_check %>% glimpse
  # data_check %>%
  #   mutate(Invoice = as.factor(Invoice),
  #          StockCode = as.factor(StockCode),
  #          InvoiceDate = as.Date(InvoiceDate),
  #          `Customer.ID` = as.factor(`Customer.ID`),
  #          Country = as.factor(Country)) %>% glimpse
  # data_check %>%
  #   mutate(Invoice = as.factor(Invoice),
  #          StockCode = as.factor(StockCode),
  #          InvoiceDate = as.Date(InvoiceDate),
  #          `Customer.ID` = as.factor(`Customer.ID`),
  #          Country = as.factor(Country)) -> data_check
  # 
  # 
  # ########################### Date
  # ###########################
  # ###########################
  # data_clean <- CustomerData # 779495*9
  # sapply(data_clean, mode)
  # 
  # # convert datetime to str
  # data_clean$InvoiceDate = as.character(data_clean$InvoiceDate)
  # # to seperate date and time
  # data_clean$date <- sapply(data_clean$InvoiceDate, FUN=\(x){strsplit(x, split=' ')[[1]][1]})
  # data_clean$time <- sapply(data_clean$InvoiceDate, FUN=\(x){strsplit(x, split=' ')[[1]][2]})
  # 
  # data_clean$year <- sapply(data_clean$date, FUN=\(x){strsplit(x, split='-')[[1]][1]})
  # data_clean$month <- sapply(data_clean$date, FUN=\(x){strsplit(x, split='-')[[1]][2]})
  # data_clean$day <- sapply(data_clean$date, FUN=\(x){strsplit(x, split='-')[[1]][3]})
  # data_clean$hour <- sapply(data_clean$time, FUN=\(x){strsplit(x, split=':')[[1]][1]})
  # 
  # data_clean$date |> as.Date("%Y-%m-%d") -> data_clean$date
  # 
  # # new variable "Day of Week" using ===> lubridate
  # # tells us the day of the week
  # data_clean$date |> wday(label=TRUE) -> data_clean$DayOfWeek
  # 
  # 
  # # new_variable "Total Price"
  # data_clean %>%
  #   mutate(Total = (Quantity-Cancelled_Q)*Price) -> data_clean
  # 
  # 
  # data_clean %>%
  #   mutate(Country = as.factor(Country),
  #          month = as.factor(month),
  #          year = as.factor(year),
  #          hour = as.factor(hour),
  #          DayOfWeek = as.factor(DayOfWeek)) -> data_clean
  # levels(data_clean$year) <- c(2009, 2010, 2011)
  # 
  # data_check = data_clean
  # 
  # ########################### Day of week analysis
  # ###########################
  # ###########################
  # 
  # # This is a summary describing what is happening on each day.
  # WeekDaySummary <- data_clean %>%
  #   group_by(date, DayOfWeek)%>%
  #   summarise(revenue = sum(Total),
  #             transactions = n_distinct(Invoice),
  #             customers = n_distinct(`Customer.ID`)) %>%
  #   mutate(avg_order_val = round(revenue/transactions, 2),
  #          avg_spend = round(revenue/customers),
  #          avg_transc = round(transactions/customers, 2)) %>%
  #   group_by(DayOfWeek) %>%
  #   mutate(MeanRev = mean(revenue),
  #          MeanTransac = mean(transactions),
  #          MeanCustomers = mean(customers),
  #          MeanAvgVal = mean(avg_order_val),
  #          MeanSpent = mean(avg_spend),
  #          MeanTranCus = mean(avg_transc))
  # # what we are having is a very large skewed bunch of data, so we will use a non-parametric test
  # # to look for statistically significant differences in our data
  # # Rank sum
  # # hypothesis: Are num of transactions in 7 days of week the same?
  # kruskal.test(transactions~DayOfWeek, data=WeekDaySummary)
  # # we have enough evidence to reject our null hypthesis
  # 
  # # hypothesis: Are num of transactions in 7 days of week the same?
  # kruskal.test(revenue~DayOfWeek, data=WeekDaySummary)
  # # we have enough evidence to reejct the null hypothesis
  # 
  # # hypothesis: Are num of transactions in 7 days of week the same?
  # kruskal.test(customers~DayOfWeek, data=WeekDaySummary)
  # # we have enough evidence to reejct the null hypothesis
  # 
  # # Now, which of our groups are diff from each other?
  # # conduct Tukey Test, PostHoc Test for anova
  # # assumptions: variance of the response variables is the same for all treatment
  # # assumptions: obervations are independent
  # anova_model = aov(transactions~DayOfWeek, data=WeekDaySummary)
  # TukeyHSD(anova_model, conf.level = 0.05)
  # # the pairwise test suggests that in 95% family-wise confidence,
  # # the transactions made in
  # 
  # ## MOn and Sun, Tue and Sun, Wed and Sun, Thu and Sun,Thu and MOn,Thu and Tue,Thu and Wed
  # 
  # # are significantly diff.
  # 
  # anova_model = aov(revenue~DayOfWeek, data=WeekDaySummary)
  # TukeyHSD(anova_model, conf.level = 0.05)
  # # the pairwise test suggests that in 95% family-wise confidence,
  # # the revenue made in
  # 
  # ## MOn and Sun, Tue and Sun, Wed and Sun, Thu and Sun, Fri and Sun
  # ## Thu and Mon,Thu and Wed
  # 
  # # are significantly diff.
  # 
  # # Which are significantly different from others
  # with(WeekDaySummary, kruskal(revenue, DayOfWeek, console=TRUE))
  # ## Thur
  # with(WeekDaySummary, kruskal(transactions, DayOfWeek, console=TRUE))
  # ## Thur and Fri
  # with(WeekDaySummary, kruskal(customers, DayOfWeek, console=TRUE))
  # ## Thur
  # 
  # ## These statistical tests tell us that mpre customers tend to shop and make
  # # more orders and spend more on Thursday. (probably for the Friday party? LAST FRIDAY NIGHT)
  # ########################### Country
  # ###########################
  # ###########################
  # # Exploring the content of variables
  # 
  # country_names <- data_clean$Country %>% unique
  # country_names %>% length
  # # there are 41 countries that made the orders
  # country_data <- as.data.frame(table(data_clean$Country))
  # country_data <- cbind(country_data, as.data.frame(country_data$Freq/sum(country_data$Freq)))
  # colnames(country_data) <- c("country", "freq", "percentage")
  # country_data$percentage |> round(4) -> country_data$percentage
  # 
  # colnames(data_clean)[7] <- "CustomerID"
  # colnames(data_clean)[17] <- "Spent"
  # 
  # data_clean %>%
  #   group_by(Country) %>%
  #   summarise(revenue=sum(Spent),
  #             customers = n_distinct(CustomerID),
  #             transactions = n_distinct(Invoice)) %>%
  #   mutate(avg_order_val = round(revenue/transactions, 2)) %>%
  #   ungroup() %>%
  #   arrange(desc(revenue)) -> CountrySummary
  # 
  # head(CountrySummary, 11)$Country[-1]
  # 
  # TopNCountries <- function(data, n, Countrysummary = CountrySummary){
  #   data %>%
  #     filter(Country %in% head(Countrysummary, n+1)$Country[-1]) %>%
  #     group_by(Country, year, month) %>%
  #     summarize(revenue = sum(Spent),
  #               transactions = n_distinct(Invoice),
  #               customers = n_distinct(CustomerID)) %>%
  #     mutate(avg_order_val = round(revenue/transactions, 2),
  #            avg_transc = round(transactions/customers, 2)) %>%
  #     ungroup() %>%
  #     arrange(desc(revenue))
  # }
  # 
  # TopNCountries(data_clean, 20) -> Top20Summary
  # TopNCountries(data_clean, 10) -> Top10Summary
  # TopNCountries(data_clean, 5) -> Top5Summary
  # 
  # data_clean$InvoiceDate %>% range()
  # 
  # country_w_year <- function(data, yvalue, ylab){
  #   title = paste0(ylab, " by Country over Time")
  #   ggplot(data, aes(x = Country, y = yvalue, fill = year)) +
  #     geom_col(position="dodge")+
  #     geom_smooth(method = 'auto', se = FALSE)+
  #     labs(x = ' Country', y = ylab, title = title)+
  #     theme(legend.position = "bottom")
  # }
  # 
  # ######################## Customer Segmentation
  # ########################
  # ########################
  # 
  # data_clean %>%
  #   group_by(CustomerID) %>%
  #   summarise(revenue = sum(Spent),
  #             transactions = n_distinct(Invoice)) %>%
  #   mutate(avg_order_val = round(revenue/transactions, 2)) %>%
  #   ungroup() %>%
  #   arrange(desc(revenue)) -> CustomerSummary
  # 
  # data_clean %>%
  #   group_by(CustomerID, Invoice) %>%
  #   summarise(revenue = sum(Spent),
  #             transactions = n_distinct(Invoice)) %>%
  #   mutate(avg_order_val = round(revenue/transactions, 2)) %>%
  #   ungroup() %>%
  #   arrange(revenue) %>%
  #   mutate(CumSum = cumsum(revenue)) -> CustomerSummary
  # 
  # CustomerSummary |> head() |> knitr::kable()
  # 
  # data_clean %>%
  #   filter(CustomerID==12380 & Spent>100) %>%
  #   transmute(desc = Description,
  #             Quantity = Quantity,
  #             Cancelled = Cancelled_Q,
  #             Spent = Spent) %>% View
  # 
  # data_clean %>%
  #   group_by(Invoice, CustomerID, Country, date, month, year, hour, DayOfWeek) %>%
  #   summarise(order_val = sum(Spent)) %>%
  #   mutate(recent = Sys.Date()-date) %>%
  #   ungroup() -> CustomerSummary
  # 
  # CustomerSummary %>% dim # 36976*10
  # 
  # CustomerSummary$recent <- as.character(CustomerSummary$recent)
  # CustomerSummary$RecentDays <- sapply(CustomerSummary$recent, FUN = \(x){
  #   strsplit(x, split=" ")[[1]][1]}) # extract 4568, remove "days"
  # CustomerSummary$RecentDays <- as.integer(CustomerSummary$RecentDays)
  # 
  # CustomerSummary %>% glimpse # 36976*11
  # 
  # CustomerSummary %>%
  #   group_by(CustomerID, Country) %>%
  #   summarize(transactions = n_distinct(Invoice),
  #             revenue = sum(order_val),
  #             MeanRev = round(mean(order_val), 2),
  #             MedianRev = median(order_val),
  #             LovesToBuyOn = names(which.max(table(DayOfWeek))),
  #             LovesToBuyAt = names(which.max(table(hour))),
  #             recency = min(RecentDays)
  #   ) %>%
  #   ungroup() -> CustomerAnalysis
  # 
  # dim(CustomerAnalysis) #5894*9
  # 
  # ###########################
  # # NLP
  # ###########################
  # new_data %>% filter(num>5) %>% arrange(desc(num)) -> new_data
  # 
  # token_words <- tokenizers::tokenize_words(new_data$Description) %>% unlist
  # token_words[grepl('^[a-zA-Z]+[a-zA-Z]$', token_words)] -> token_words
  # 
  # doc = paste(token_words, collapse = " ")
  # doc <- Corpus(VectorSource(doc))
  # 
  # # convert all text into lower case
  # doc <- tm_map(doc, content_transformer(tolower))
  # # remove numbers
  # doc <- tm_map(doc, removeNumbers)
  # # remove common stopwords
  # doc <- tm_map(doc, removeWords, stopwords("english"))
  # # no punctuations need to be removed
  # doc <- tm_map(doc, stripWhitespace)
  # # stemming is not necessary either
  # #doc <- tm_map(doc, stemDocument)
  # 
  # # term document matrix
  # TermDocumentMatrix(doc) -> tf_tdm
  # tf_tdm %>% as.matrix -> tf
  # tf <- sort(rowSums(tf), decreasing = TRUE)
  # tf <- data.frame(word=names(tf), freq=tf)
  # 
  # tf %>% filter(!(tf$word %in% c("red",
  #                                "pink",
  #                                "orange",
  #                                "green",
  #                                "blue",
  #                                "tag",
  #                                "white",
  #                                "redwhite"))) -> tf
  # 
  # head(tf, 30)
  # 
  # # generate wc
  # set.seed(42)
  # wordcloud(words = tf$word, freq = tf$freq, min.freq = 1,
  #           max.words=200, random.order=FALSE, rot.per=0.35,
  #           colors=brewer.pal(8, "Dark2"))
  # 
  # # the above wc clearly shows that "set", "heart", "bag ,"glass"
  # # and "box" are the 5 most common words appears in the product description
  # 
  # findFreqTerms(tf_tdm, lowfreq = 1)
  ############################################################code_run_before#####################################################
  output$missing_table <- renderDataTable({
    datatable(as.data.frame(sapply(year2009_2010_show, \(x) {sum(is.na(x))})))
  })
  output$missing_table_1 <- renderDataTable({
    datatable(as.data.frame(sapply(year2010_2011_show, \(x) {sum(is.na(x))})))
  })
  output$data_info <- renderDataTable({
    datatable(Info_table)
  })
  output$product_info <- renderDataTable({
    datatable(Info_customer_product)
  })
  output$product_num <- renderDataTable({
    datatable(head(country_table_show, 10))
  })
  output$top_customer <- renderDataTable({
    datatable(transaction_country(CustomerData_show,50))
  })
  output$product_purchase_once <- renderDataTable({
    CustomerData[never_come_back,] %>%
      group_by(Description) %>%
      summarise(Quantity = sum(Quantity),
                Price = mean(Price)) %>%
      arrange(desc(Quantity)) |>
      head(30) %>% 
      datatable()
  })
  output$cancel_order <-renderDataTable({
    datatable(cancel_order_show)
  })
  output$cancel_check <- renderPrint({
    
    for(row in row.names(data_check_use)){
      customerID <- data_check[row, ]$`Customer.ID`
      descrip <- data_check[row, ]$Description
      quantity <- -data_check[row, ]$Quantity
      size = CustomerData %>% filter(`Customer.ID` == customerID,
                                     Description == descrip,
                                     Quantity == quantity) %>% dim
      if (size[1]==0){
        cat(paste(data_check[row,], '\n'))
        cat("-----> Hypothesis not fulfilled.")
        break
      }
    }
    
    # We found that the initial hypothesis is not verified. 
    # (ie. cancellations do not necessarily correspond to orders that would have
    # made afterwards.)
  })
  output$cancel_country <- renderDataTable({
      datatable(cancel_country_show)
    # Apart from the UK, Germany has the most cancellations country of order
  })
  output$doubtful_cleaning <- renderPrint({
    doubtful_cleaning_show %>% glimpse #779495*9
  })
  # output$stock_code <- renderDataTable({
  #   special_codes %>% 
  #     group_by(StockCode, Description) %>%
  #     summarise(num = n()) %>%
  #     datatable()
  #   # several types of peculiar transactions connected to post charge or bank charges.
  # })
  output$check_consume_range <- renderDataTable({
    data_check %>% 
      group_by(`Customer.ID`, Invoice, Price) %>%
      summarize(Total_price = (Quantity-Cancelled_Q)*Price,
                Date = mean(InvoiceDate)) -> new_data
    
    datatable(head(new_data))
  })
  output$recoding <- function(){
    ## recoding and character variables to factors
    data_check %>% glimpse
    data_check %>%
      mutate(Invoice = as.factor(Invoice),
             StockCode = as.factor(StockCode),
             InvoiceDate = as.Date(InvoiceDate),
             `Customer.ID` = as.factor(`Customer.ID`),
             Country = as.factor(Country)) %>% glimpse
    data_check %>%
      mutate(Invoice = as.factor(Invoice),
             StockCode = as.factor(StockCode),
             InvoiceDate = as.Date(InvoiceDate),
             `Customer.ID` = as.factor(`Customer.ID`),
             Country = as.factor(Country)) -> data_check
  }
  output$k_test <- renderPrint({
    # what we are having is a very large skewed bunch of data, so we will use a non-parametric test
    # to look for statistically significant differences in our data
    # Rank sum
    # hypothesis: Are num of transactions in 7 days of week the same?
    kruskal.test(transactions~DayOfWeek, data=WeekDaySummary)
    # we have enough evidence to reject our null hypthesis
    
    # hypothesis: Are num of transactions in 7 days of week the same?
    kruskal.test(revenue~DayOfWeek, data=WeekDaySummary)
    # we have enough evidence to reejct the null hypothesis
    
    # hypothesis: Are num of transactions in 7 days of week the same?
    kruskal.test(customers~DayOfWeek, data=WeekDaySummary)
    # we have enough evidence to reejct the null hypothesis
  })
  output$tukey <- renderPrint({
    # Now, which of our groups are diff from each other?
    # conduct Tukey Test, PostHoc Test for anova
    # assumptions: variance of the response variables is the same for all treatment
    # assumptions: obervations are independent
    anova_model = aov(transactions~DayOfWeek, data=WeekDaySummary)
    TukeyHSD(anova_model, conf.level = 0.05)
    # the pairwise test suggests that in 95% family-wise confidence,
    # the transactions made in
    
    ## MOn and Sun, Tue and Sun, Wed and Sun, Thu and Sun,Thu and MOn,Thu and Tue,Thu and Wed
    
    # are significantly diff.
    
    anova_model = aov(revenue~DayOfWeek, data=WeekDaySummary)
    TukeyHSD(anova_model, conf.level = 0.05)
    # the pairwise test suggests that in 95% family-wise confidence,
    # the revenue made in 
    
    ## MOn and Sun, Tue and Sun, Wed and Sun, Thu and Sun, Fri and Sun
    ## Thu and Mon,Thu and Wed
    
    # are significantly diff.
    # Which are significantly different from others
    with(WeekDaySummary, kruskal(revenue, DayOfWeek, console=TRUE))
    ## Thur
    with(WeekDaySummary, kruskal(transactions, DayOfWeek, console=TRUE))
    ## Thur and Fri
    with(WeekDaySummary, kruskal(customers, DayOfWeek, console=TRUE))
    ## Thur
    
    ## These statistical tests tell us that mpre customers tend to shop and make 
    # more orders and spend more on Thursday. (probably for the Friday party? LAST FRIDAY NIGHT)
  })
  
  output$country_data <- renderDataTable({
    # Exploring the content of variables
    country_names <- data_clean$Country %>% unique 
    country_names %>% length
    # there are 41 countries that made the orders
    country_data <- as.data.frame(table(data_clean$Country))
    country_data <- cbind(country_data, as.data.frame(country_data$Freq/sum(country_data$Freq)))
    colnames(country_data) <- c("country", "freq", "percentage")
    country_data$percentage |> round(4) -> country_data$percentage
    datatable(country_data)
    # We can observe from the table that the data set is largely dominated by orders
    # made from the UK.
  })
  output$country_view <- renderDataTable({
    # most of the transactions are made in Europe
    colnames(data_clean)[7] <- "CustomerID"
    colnames(data_clean)[17] <- "Spent"
    
    data_clean %>%
      group_by(Country) %>%
      summarise(revenue=sum(Spent),
                customers = n_distinct(CustomerID),
                transactions = n_distinct(Invoice)) %>%
      mutate(avg_order_val = round(revenue/transactions, 2)) %>%
      ungroup() %>%
      arrange(desc(revenue)) -> CountrySummary
    
    head(CountrySummary) %>% datatable()
  })
  output$top5country_summary <- renderDataTable({
    # a lot of different countries are contributing a good amount of revenue,
    # we exclude UK in the above visualization as we know that this is a UK retailer
    # so improving UK performance is undoubtly being on the radar
    # dig in to the top 10 countries in terms of the revenue contribution
    
    head(CountrySummary, 11)$Country[-1]
    
    TopNCountries <- function(data, n, Countrysummary = CountrySummary){
      data %>% 
        filter(Country %in% head(Countrysummary, n+1)$Country[-1]) %>%
        group_by(Country, year, month) %>%
        summarize(revenue = sum(Spent),
                  transactions = n_distinct(Invoice),
                  customers = n_distinct(CustomerID)) %>%
        mutate(avg_order_val = round(revenue/transactions, 2),
               avg_transc = round(transactions/customers, 2)) %>%
        ungroup() %>%
        arrange(desc(revenue))
    }
    TopNCountries(data_clean, 5) -> Top5Summary
    
    datatable(head(Top5Summary))
  })
  output$customer_eg <- renderDataTable({
    data_clean %>%
      group_by(CustomerID, Invoice) %>%
      summarise(revenue = sum(Spent),
                transactions = n_distinct(Invoice)) %>%
      mutate(avg_order_val = round(revenue/transactions, 2)) %>%
      ungroup() %>%
      arrange(revenue) %>%
      mutate(CumSum = cumsum(revenue)) -> CustomerSummary
    
    CustomerSummary |> head() |> knitr::kable()
    
    data_clean %>% 
      filter(CustomerID==12380 & Spent>100) %>%
      transmute(desc = Description,
                Quantity = Quantity,
                Cancelled = Cancelled_Q,
                Spent = Spent) %>% datatable()
    
    # From this example, I think CustomerID 12380 is a woman, she 
    # probably has a cat, loves red, and maybe a big fan of russian doll (jk).
    ## Weirdly, she spent alot on POSTAGE??
  })
  output$customer_analysis <- renderPrint({
    ## So, Where did this customer come from?
    ## when did this person make a purchase?
    
    data_clean %>%
      group_by(Invoice, CustomerID, Country, date, month, year, hour, DayOfWeek) %>%
      summarise(order_val = sum(Spent)) %>%
      mutate(recent = Sys.Date()-date) %>%
      ungroup() -> CustomerSummary
    
    CustomerSummary %>% dim # 36976*10
    
    CustomerSummary$recent <- as.character(CustomerSummary$recent)
    CustomerSummary$RecentDays <- sapply(CustomerSummary$recent, FUN = \(x){
      strsplit(x, split=" ")[[1]][1]}) # extract 4568, remove "days"
    CustomerSummary$RecentDays <- as.integer(CustomerSummary$RecentDays)
    
    CustomerSummary %>% glimpse # 36976*11
    
    CustomerSummary %>%
      group_by(CustomerID, Country) %>%
      summarize(transactions = n_distinct(Invoice),
                revenue = sum(order_val),
                MeanRev = round(mean(order_val), 2),
                MedianRev = median(order_val),
                LovesToBuyOn = names(which.max(table(DayOfWeek))),
                LovesToBuyAt = names(which.max(table(hour))),
                recency = min(RecentDays)
      ) %>%
      ungroup() -> CustomerAnalysis
    
    dim(CustomerAnalysis) #5894*9
    
    knitr::kable(head(CustomerAnalysis))
    CustomerAnalysis %>%
      filter(transactions>=1, revenue>=50) %>% glimpse# 5799*9
    # This is a data frame describing how many orders, revenue, and average order value
    # as well as the day of the week and the time of the day they most frequently place
    # orders.
  })
  output$nlp_head <-renderDataTable({
    new_data %>% filter(num>5) %>% arrange(desc(num)) -> new_data
    
    token_words <- tokenizers::tokenize_words(new_data$Description) %>% unlist
    token_words[grepl('^[a-zA-Z]+[a-zA-Z]$', token_words)] -> token_words
    
    doc = paste(token_words, collapse = " ")
    doc <- Corpus(VectorSource(doc))
    
    # convert all text into lower case
    doc <- tm_map(doc, content_transformer(tolower))
    # remove numbers
    doc <- tm_map(doc, removeNumbers)
    # remove common stopwords
    doc <- tm_map(doc, removeWords, stopwords("english"))
    # no punctuations need to be removed
    doc <- tm_map(doc, stripWhitespace)
    # stemming is not necessary either
    #doc <- tm_map(doc, stemDocument)
    
    # term document matrix
    TermDocumentMatrix(doc) -> tf_tdm
    tf_tdm %>% as.matrix -> tf
    tf <- sort(rowSums(tf), decreasing = TRUE)
    tf <- data.frame(word=names(tf), freq=tf)
    
    tf %>% filter(!(tf$word %in% c("red",
                                   "pink", 
                                   "orange",
                                   "green",
                                   "blue",
                                   "tag",
                                   "white",
                                   "redwhite"))) -> tf
    
    datatable(head(tf, 30))
  })
  wordcloud_rep <- repeatable(wordcloud)
  output$plot <- renderPlot({
    wordcloud_rep(words = tf$word,freq = tf$freq,min.freq = input$freq, max.words=input$max,
                  random.order=FALSE, rot.per=0.35, colors=brewer.pal(8, "Dark2"))
  })
}

shinyApp(ui, server)