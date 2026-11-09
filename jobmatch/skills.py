"""Từ điển kỹ năng = skill tag ITviec + bảng alias (Việt/Anh, viết tắt, tên gọi khác).

    ex = SkillExtractor.from_jobs(jobs)
    ex.extract("Có kinh nghiệm PySpark, học máy, sklearn")  → {"Apache Spark", "Machine Learning", "Scikit-learn"}
    canonical("pyspark")                                     → "Apache Spark"
"""
import re
from collections import Counter

# canonical → các cách viết khác (không phân biệt hoa thường, trừ CASE_SENSITIVE)
ALIASES = {
    # --- Ngôn ngữ lập trình ---
    "Python": ["python", "python3"],
    "SQL": ["sql", "t-sql", "tsql", "pl/sql", "plsql", "spark sql", "sparksql"],
    "Java": ["java"],
    "Scala": ["scala"],
    "C++": ["c++", "cpp", "c/c++"],
    "C#": ["c#", "csharp"],
    "Golang": ["golang", "go lang", "go language", "go (golang)"],
    "R": ["r language", "r programming", "ngôn ngữ r", "rstudio", "r studio"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript"],
    "NodeJS": ["node.js", "nodejs", "node js"],
    "Rust": ["rust"],
    "Bash": ["bash", "shell script", "shell scripting", "shell"],
    "MATLAB": ["matlab"],
    # --- AI / ML ---
    "AI": ["AI", "artificial intelligence", "trí tuệ nhân tạo"],
    "Generative AI": ["generative ai", "genai", "gen ai", "ai tạo sinh"],
    "Agentic AI": ["agentic ai", "ai agent", "ai agents", "multi-agent", "agentic"],
    "Machine Learning": ["machine learning", "ML", "học máy"],
    "Deep Learning": ["deep learning", "học sâu", "neural network", "neural networks", "mạng nơ-ron", "mạng neural"],
    "NLP": ["nlp", "natural language processing", "xử lý ngôn ngữ tự nhiên", "text mining"],
    "Computer Vision": ["computer vision", "thị giác máy tính", "image processing", "xử lý ảnh",
                        "object detection", "image classification", "image segmentation"],
    "LLM": ["llm", "llms", "large language model", "large language models", "mô hình ngôn ngữ lớn",
            "gpt", "chatgpt", "openai api", "gemini api", "llama"],
    "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "Prompt Engineering": ["prompt engineering", "prompting"],
    "LangChain": ["langchain", "langgraph", "llamaindex", "llama index", "llama-index"],
    "Fine-tuning": ["fine-tuning", "fine tuning", "finetuning", "fine-tune", "lora", "qlora", "peft"],
    "Transformers": ["transformers", "transformer", "hugging face", "huggingface", "bert", "t5"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow", "tf2", "keras"],
    "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "XGBoost": ["xgboost", "lightgbm", "catboost", "gradient boosting"],
    "OpenCV": ["opencv", "open cv"],
    "YOLO": ["yolo", "yolov5", "yolov8"],
    "Speech Processing": ["speech recognition", "asr", "text-to-speech", "tts", "speech processing", "nhận dạng giọng nói"],
    "Recommender Systems": ["recommender system", "recommendation system", "recommender systems",
                            "recommendation systems", "hệ thống gợi ý", "collaborative filtering"],
    "Reinforcement Learning": ["reinforcement learning", "học tăng cường"],
    "Time Series": ["time series", "time-series", "chuỗi thời gian", "forecasting"],
    "MLOps": ["mlops", "ml ops", "mlflow", "kubeflow", "model deployment", "model serving", "bentoml", "triton"],
    "Vector Database": ["vector database", "vector db", "faiss", "pinecone", "milvus", "qdrant", "chromadb",
                        "chroma", "weaviate", "pgvector"],
    "Statistics": ["statistics", "statistical", "thống kê", "xác suất thống kê", "probability", "hypothesis testing",
                   "a/b testing", "ab testing"],
    "Data Science": ["data science", "khoa học dữ liệu", "data scientist"],
    # --- Data engineering ---
    "Apache Spark": ["spark", "pyspark", "apache spark", "spark streaming"],
    "Hadoop": ["hadoop", "hdfs", "mapreduce", "yarn"],
    "Hive": ["hive", "hiveql", "impala"],
    "Apache Kafka": ["kafka", "apache kafka", "kafka streams"],
    "Apache Flink": ["flink", "apache flink"],
    "Apache Airflow": ["airflow", "apache airflow", "dagster", "prefect"],
    "dbt": ["dbt", "data build tool"],
    "ETL": ["etl", "elt", "etl/elt", "data pipeline", "data pipelines", "pipeline dữ liệu"],
    "Data Warehouse": ["data warehouse", "data warehousing", "kho dữ liệu", "dwh", "data mart", "dimensional modeling",
                       "star schema", "kimball"],
    "Data Lake": ["data lake", "lakehouse", "delta lake", "iceberg", "apache iceberg", "apache hudi"],
    "Data Modeling": ["data modeling", "data modelling", "mô hình hoá dữ liệu"],
    "Big Data": ["big data", "dữ liệu lớn"],
    "Databricks": ["databricks"],
    "Snowflake": ["snowflake"],
    "BigQuery": ["bigquery", "big query"],
    "Redshift": ["redshift", "aws redshift"],
    "Trino": ["trino", "presto", "athena"],
    "ClickHouse": ["clickhouse"],
    "NiFi": ["nifi", "apache nifi"],
    "Debezium": ["debezium", "cdc", "change data capture"],
    "Data Quality": ["data quality", "great expectations", "data governance", "data lineage"],
    "Streaming": ["stream processing", "real-time processing", "realtime processing", "streaming data"],
    # --- Database ---
    "PostgreSQL": ["postgresql", "postgres", "postgre"],
    "MySQL": ["mysql", "mariadb"],
    "SQL Server": ["sql server", "mssql", "ms sql"],
    "Oracle": ["oracle", "oracle db"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "Elasticsearch": ["elasticsearch", "elastic search", "elk", "opensearch"],
    "Cassandra": ["cassandra"],
    "NoSQL": ["nosql"],
    "Database": ["database", "databases", "cơ sở dữ liệu", "rdbms"],
    # --- Cloud / DevOps ---
    "AWS": ["aws", "amazon web services", "s3", "ec2", "aws glue", "aws lambda", "sagemaker", "emr", "kinesis"],
    "GCP": ["gcp", "google cloud", "google cloud platform", "vertex ai", "dataflow", "dataproc", "pub/sub"],
    "Azure": ["azure", "microsoft azure", "azure data factory", "adf", "synapse", "azure ml"],
    "Cloud": ["cloud", "cloud computing", "điện toán đám mây"],
    "Docker": ["docker", "container", "containers", "docker compose", "docker-compose"],
    "Kubernetes": ["kubernetes", "k8s", "helm", "eks", "gke", "aks"],
    "Linux": ["linux", "ubuntu", "unix", "centos"],
    "Git": ["git", "github", "gitlab", "bitbucket"],
    "CI/CD": ["ci/cd", "cicd", "ci cd", "jenkins", "github actions", "gitlab ci"],
    "DevOps": ["devops"],
    "Terraform": ["terraform", "infrastructure as code", "iac"],
    # --- Backend / API ---
    "FastAPI": ["fastapi", "fast api"],
    "Flask": ["flask"],
    "Django": ["django"],
    "RESTful API": ["rest api", "restful", "restful api", "rest apis", "api"],
    "Microservices": ["microservices", "microservice", "micro-services"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Jupyter": ["jupyter", "jupyter notebook"],
    # --- BI / phân tích ---
    "Power BI": ["power bi", "powerbi", "dax"],
    "Tableau": ["tableau"],
    "Looker": ["looker", "looker studio", "data studio"],
    "Excel": ["excel", "ms excel", "google sheets"],
    "Data Analysis": ["data analysis", "phân tích dữ liệu", "data analytics", "data analyst", "eda",
                      "exploratory data analysis"],
    "Data Visualization": ["data visualization", "data visualisation", "trực quan hoá dữ liệu", "matplotlib",
                           "seaborn", "plotly"],
    "BI": ["BI", "business intelligence"],
    # --- Kỹ năng khác ---
    "English": ["english", "tiếng anh", "ielts", "toeic", "toefl"],
    "Japanese": ["japanese", "tiếng nhật", "jlpt"],
    "Agile": ["agile", "scrum", "kanban"],
    "Algorithms": ["algorithms", "data structures", "cấu trúc dữ liệu", "giải thuật", "thuật toán", "leetcode"],
    "OOP": ["oop", "object-oriented", "object oriented", "hướng đối tượng"],
}

# Viết tắt dễ trùng từ thường (vd. "ai" = "ai" trong tiếng Việt) → chỉ khớp đúng hoa/thường
CASE_SENSITIVE = {"AI", "ML", "BI", "ADF", "EMR", "ASR", "TTS", "IaC", "CDC", "ELK", "EDA", "T5", "DWH"}
# Alias mơ hồ quá → không dùng khi quét văn bản tự do (vẫn dùng để chuẩn hoá tag)
NO_FREE_TEXT = {"api", "shell", "container", "containers", "cloud", "torch", "transformer", "chroma", "yarn",
                "prompting", "database", "databases", "statistical", "forecasting", "oracle", "go", "r", "c"}

# Tag ITviec là tên vị trí chứ không phải kỹ năng → không tính vào kỹ năng của tin
NON_SKILL_TAGS = {"Data Engineer", "Fullstack", "Tester", "QA QC", "Product Owner", "Team Management",
                  "Data Scientist", "Backend Developer", "Frontend Developer", "Team Leader", "Manager",
                  "Solution Architecture", "Software Architecture"}

_LOOKUP = {}
for _canon, _als in ALIASES.items():
    _LOOKUP[_canon.lower()] = _canon
    for _a in _als:
        _LOOKUP.setdefault(_a.lower(), _canon)


def canonical(name):
    """Chuẩn hoá tên kỹ năng về dạng gốc; tên lạ thì giữ nguyên (bỏ khoảng trắng thừa)."""
    s = re.sub(r"\s+", " ", (name or "").strip())
    return _LOOKUP.get(s.lower(), s)


def canonical_set(names):
    return {c for c in (canonical(n) for n in names if n and str(n).strip()) if c not in NON_SKILL_TAGS}


# Alias tiếng Việt nằm trong cụm từ khác nghĩa: "Khoa học máy tính" ≠ "học máy"
NOT_FOLLOWED_BY = {"học máy": r"\s+tính", "thống kê": r"\s+(?:cho thấy|được)"}


def _pattern(alias):
    # "machine learning" khớp cả "machine-learning", "machine_learning", "Machine  Learning"
    body = r"[\s\-_]*".join(re.escape(tok) for tok in re.split(r"[\s\-]+", alias.strip()) if tok)
    neg = NOT_FOLLOWED_BY.get(alias.lower())
    return rf"(?<![\w+#]){body}(?![\w+#])" + (rf"(?!{neg})" if neg else "")


class SkillExtractor:
    def __init__(self, extra_skills=()):
        self.aliases = {k: list(v) for k, v in ALIASES.items()}
        for s in extra_skills:              # tag ITviec chưa có trong bảng → tự làm alias của chính nó
            c = canonical(s)
            if c not in self.aliases and c not in NON_SKILL_TAGS:
                self.aliases[c] = [c]
        self._rules = []
        cs_form = {x.lower(): x for x in CASE_SENSITIVE}
        for canon, als in self.aliases.items():
            terms = {canon, *als}
            for t in terms:
                if t.lower() in NO_FREE_TEXT or len(t) < 2:
                    continue
                if t.lower() in cs_form:
                    rx = re.compile(_pattern(cs_form[t.lower()]))
                elif len(t) <= 3 and t.isupper() and t.lower() not in _LOOKUP:  # tag lạ viết tắt, vd "QA"
                    rx = re.compile(_pattern(t))
                else:
                    rx = re.compile(_pattern(t), re.IGNORECASE)
                self._rules.append((rx, canon))

    @classmethod
    def from_jobs(cls, jobs):
        return cls(s for j in jobs for s in j.get("skills", []))

    def extract(self, text):
        text = (text or "").replace("’", "'")
        return {canon for rx, canon in self._rules if rx.search(text)}

    def vocabulary(self):
        return sorted(self.aliases)

    def mentioned(self, skill, text):
        """Kỹ năng (hoặc 1 alias của nó) có thật sự xuất hiện trong văn bản không — dùng đo grounding."""
        canon = canonical(skill)
        terms = {skill, canon, *self.aliases.get(canon, [])}
        return any(re.search(_pattern(t), text or "", re.IGNORECASE) for t in terms if len(t) >= 2)


def skill_counts(jobs, field="job_skills"):
    return Counter(s for j in jobs for s in j.get(field, []))
