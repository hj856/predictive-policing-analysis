import os
import re
import jieba
import jieba.analyse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from wordcloud import WordCloud
from gensim import corpora, models

# -------------------------- 全局配置 --------------------------
# 目录配置
ROOT_DATA_DIR = "data"
RESULT_DIR = "result"
# 停用词扩展（可根据需求补充）
STOPWORDS_CN = {"的", "了", "是", "在", "和", "有", "我", "你", "他", "都", "也", "就", "而", "及", "与", "对", "对于", "关于", "则", "但",
                "若", "如", "将", "为", "之", "其", "所", "以", "于", "即", "因", "由", "从", "到", "等", "可", "能", "会", "应", "该", "这",
                "那", "此", "彼", "个", "件", "项", "条", "课题"}
STOPWORDS_EN = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "in", "on", "at", "for", "of", "to",
                "with", "by", "this", "that", "these", "those", "it", "he", "she", "we", "they", "will", "would", "can",
                "could", "should", "may", "might", "such", "any", "shall", "b", "not", "s", "as", "from", "other", "be",
                "including"}
# 绘图配置（解决中文显示问题，优化美观度）
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial"]  # 中英文字体
plt.rcParams["axes.unicode_minus"] = False  # 显示负号
plt.rcParams["figure.facecolor"] = "white"  # 画布背景色


# -------------------------- 步骤1：读取并合并文本 --------------------------
def read_and_merge_texts():
    """
    读取data下的子文件夹，合并同文件夹内所有txt文件为单一文本
    返回：字典 {"cn": 合并后的中文政策文本, "en": 合并后的英文政策文本}
    """
    # 创建结果目录
    os.makedirs(RESULT_DIR, exist_ok=True)

    merged_text = {"cn": "", "en": ""}
    folder_lang_map = {
        "cn_policy": "cn",
        "us_policy": "en"
    }

    # 遍历子文件夹，合并所有txt文件
    for folder_name, lang in folder_lang_map.items():
        folder_path = os.path.join(ROOT_DATA_DIR, folder_name)
        if not os.path.exists(folder_path):
            print(f"⚠️  未找到 {folder_path} 文件夹，该语言无文本数据")
            continue

        # 合并该文件夹下所有txt文件内容
        merged_content = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(folder_path, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    merged_content.append(content)
                except UnicodeDecodeError:
                    with open(file_path, "r", encoding="gbk") as f:
                        content = f.read().strip()
                    merged_content.append(content)
                except Exception as e:
                    print(f"❌ 读取 {file_path} 失败：{e}")

        # 合并为单一文本（用换行分隔不同文件内容）
        merged_text[lang] = "\n".join(merged_content).strip()
        print(f"✅ {folder_name} 文件夹合并完成，总文本长度：{len(merged_text[lang])} 字符")

    return merged_text


# -------------------------- 步骤2：清洗+分词 --------------------------
def clean_and_tokenize(merged_text):
    """
    清洗并分词合并后的总文本
    参数：merged_text - 步骤1返回的合并文本字典
    返回：字典 {"cn": 中文分词列表, "en": 英文分词列表}
    """
    tokenized_data = {"cn": [], "en": []}

    for lang, content in merged_text.items():
        if not content:
            print(f"⚠️  {lang.upper()} 合并后文本为空，跳过分词")
            continue

        # 清洗：去除特殊字符、空白、多余换行
        content = re.sub(r"[^\w\s]", " ", content)  # 保留字母/数字/空格
        content = re.sub(r"\s+", " ", content).strip()

        # 分词+去停用词
        if lang == "cn":
            words = jieba.lcut(content)
            # 过滤停用词、单字、纯数字
            clean_words = [w for w in words if w not in STOPWORDS_CN and len(w) > 1 and not w.isdigit()]
        else:  # en
            words = content.lower().split()
            # 过滤停用词、非字母字符
            clean_words = [w.strip(".,!?;:()[]\"'") for w in words if w not in STOPWORDS_EN and w.isalpha()]

        tokenized_data[lang] = clean_words
        print(f"✅ {lang.upper()} 分词完成，有效词汇数量：{len(tokenized_data[lang])}")

    return tokenized_data


# -------------------------- 步骤3：词频+词云 --------------------------
def word_freq_and_wordcloud(tokenized_data):
    """
    对合并后的分词结果统计词频、生成词云，保存到result目录
    参数：tokenized_data - 步骤2返回的分词字典
    """
    for lang, words in tokenized_data.items():
        if not words:
            print(f"⚠️  {lang.upper()} 无有效词汇，跳过词频/词云分析")
            continue

        # 1. 统计词频并保存CSV
        freq = Counter(words)
        freq_df = pd.DataFrame(freq.most_common(30), columns=["词汇", "频次"])
        freq_save_path = os.path.join(RESULT_DIR, f"{lang}_policy_词频表.csv")
        freq_df.to_csv(freq_save_path, index=False, encoding="utf-8-sig")
        print(f"✅ {lang.upper()} 词频表已保存：{freq_save_path}")

        # 2. 生成词云并保存
        try:
            wc = WordCloud(
                width=1000, height=800,
                background_color="white",
                font_path="simhei.ttf" if lang == "cn" else None,  # 中文指定字体
                max_words=150,
                max_font_size=200,
                colormap="viridis"
            )
            wc.generate(" ".join(words))

            # 保存词云图片
            wc_save_path = os.path.join(RESULT_DIR, f"{lang}_policy_词云.png")
            wc.to_file(wc_save_path)
            print(f"✅ {lang.upper()} 词云已保存：{wc_save_path}")
        except Exception as e:
            print(f"❌ 生成 {lang.upper()} 词云失败：{e}")


# -------------------------- 步骤4：学术极简版 · 关键词共现网络 --------------------------
def keyword_cooccurrence_network(tokenized_data):
    for lang, words in tokenized_data.items():
        if len(words) < 50:
            print(f"⚠️ {lang.upper()} 词汇不足，跳过共现网络")
            continue

        # ===================== 1. 只保留最核心的高频词（控制节点数量）=====================
        word_freq = Counter(words)
        top_nodes = 25  # 最多25个词，学术图最干净
        top_words = {w for w, _ in word_freq.most_common(top_nodes)}
        filtered = [w for w in words if w in top_words]

        # ===================== 2. 统计共现（窗口小，更严谨）=====================
        window = 4
        co_occur = Counter()
        for i in range(len(filtered) - window + 1):
            win = filtered[i:i+window]
            for a, b in zip(win, win[1:]):
                if a != b:
                    co_occur[tuple(sorted((a, b)))] += 1

        # ===================== 3. 只保留高共现强度（彻底消灭乱线）=====================
        min_weight = 5
        strong_edges = {(a, b): w for (a, b), w in co_occur.items() if w >= min_weight}

        if not strong_edges:
            print(f"⚠️ {lang.upper()} 无强共现边，跳过绘图")
            continue

        # ===================== 4. 构建干净网络图 =====================
        G = nx.Graph()
        for (a, b), w in strong_edges.items():
            G.add_edge(a, b, weight=w)

        # 只保留最大连通子图（学术图标准做法）
        if len(list(nx.connected_components(G))) > 0:
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()

        # ===================== 5. 学术风格绘图 =====================
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.figure(figsize=(10, 8), dpi=300)

        # 布局：学术图最常用 = 力导向布局（干净不散乱）
        pos = nx.spring_layout(G, seed=42, k=3.5, iterations=100)

        # 节点：大小 = 词频（学术规范）
        node_size = [word_freq[node] * 35 for node in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos, node_size=node_size,
             alpha=0.85, linewidths=1, edgecolors="white"
        )

        # 边：只画强边，不杂乱
        edge_width = [0.6 + G[u][v]["weight"] * 0.12 for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos, width=edge_width, alpha=0.4, edge_color="#999999"
        )

        # 标签：清晰、不重叠、学术字体
        nx.draw_networkx_labels(
            G, pos, font_size=10, font_family="SimHei", font_weight="normal",
            font_color="#101010"
        )

        plt.axis("off")
        plt.tight_layout()

        # 保存
        path = os.path.join(RESULT_DIR, f"{lang}_policy_co_network_academic.png")
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"✅ 学术版共现网络已保存：{path}")
# -------------------------- 步骤5：LDA主题模型 --------------------------
def lda_topic_analysis(tokenized_data):
    """
    对合并后的分词结果训练LDA主题模型，保存主题结果
    参数：tokenized_data - 步骤2返回的分词字典
    """
    for lang, words in tokenized_data.items():
        if len(words) < 50:  # 词汇量过少则跳过
            print(f"⚠️  {lang.upper()} 有效词汇不足50个，跳过LDA训练")
            continue

        # 构建LDA输入：将整体分词列表拆分为多个"文档"（按固定长度切分）
        doc_length = 100  # 每个虚拟文档100个词
        docs = [words[i:i + doc_length] for i in range(0, len(words), doc_length) if len(words[i:i + doc_length]) >= 10]
        if len(docs) < 3:
            print(f"⚠️  {lang.upper()} 虚拟文档数量不足，跳过LDA训练")
            continue

        # 构建词典和语料库
        dictionary = corpora.Dictionary(docs)
        dictionary.filter_extremes(no_below=2, no_above=0.8)  # 过滤极端词汇
        corpus = [dictionary.doc2bow(doc) for doc in docs]

        # 训练LDA模型（设置4个主题，可调整）
        num_topics = 4
        lda = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=100,
            passes=20,
            alpha="auto",
            eta="auto"
        )

        # 提取主题结果并保存
        topics = lda.print_topics(num_words=10)
        lda_save_path = os.path.join(RESULT_DIR, f"{lang}_policy_LDA主题结果.txt")
        with open(lda_save_path, "w", encoding="utf-8") as f:
            f.write(f"{lang.upper()} 政策文本LDA主题模型结果（{num_topics}个主题）\n")
            f.write("=" * 60 + "\n\n")
            for idx, topic in topics:
                # 格式化主题结果（更易读）
                topic_words = [item.split("*")[1].replace('"', '').strip() for item in topic.split(" + ")]
                f.write(f"主题 {idx + 1}：{' | '.join(topic_words)}\n")
                f.write(f"原始权重：{topic}\n\n")

        # 打印主题结果
        print(f"\n📊 {lang.upper()} LDA主题结果：")
        for idx, topic in topics:
            print(f"主题 {idx + 1}：{topic}")
        print(f"✅ {lang.upper()} LDA结果已保存：{lda_save_path}")


# -------------------------- 步骤6：主函数 --------------------------
def main():
    """主函数：串联所有步骤，合并文本后整体分析"""
    print("🚀 开始政策文本合并分析流程...\n")

    # 步骤1：读取并合并文本
    print("【步骤1】读取并合并文件夹内所有文本...")
    merged_text = read_and_merge_texts()
    if not merged_text["cn"] and not merged_text["en"]:
        print("❌ 未读取到任何文本数据，程序终止")
        return

    # 步骤2：清洗+分词
    print("\n【步骤2】文本清洗与分词...")
    tokenized_data = clean_and_tokenize(merged_text)

    # 步骤3：词频统计+词云生成
    print("\n【步骤3】词频统计与词云生成...")
    word_freq_and_wordcloud(tokenized_data)

    # 步骤4：关键词共现网络（优化版）
    print("\n【步骤4】构建优化版关键词共现网络...")
    keyword_cooccurrence_network(tokenized_data)

    # 步骤5：LDA主题模型
    print("\n【步骤5】训练LDA主题模型...")
    lda_topic_analysis(tokenized_data)

    print("\n🎉 所有分析完成！结果已全部保存到 result 目录。")


if __name__ == "__main__":
    # 安装依赖（首次运行取消注释执行）
    # os.system("pip install jieba pandas numpy matplotlib wordcloud gensim networkx")

    # 运行主函数
    main()