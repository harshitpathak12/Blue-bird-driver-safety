"""MongoDB client — singleton connection manager."""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from configs.config_loader import ConfigLoader
from utils.logger import get_logger

logger = get_logger(__name__)


class MongoDBClient:
    """Encapsulates MongoDB connection and collection handles."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        config = ConfigLoader.load()
        self._mongo_url = config["mongodb"]["url"]
        self._db_name = config["mongodb"]["database"]
        self._client = MongoClient(self._mongo_url, serverSelectionTimeoutMS=5000)
        self._db = self._client[self._db_name]
        self._initialized = True

        try:
            self._client.admin.command("ping")
            self._db.list_collection_names()
            logger.info("MongoDB connected: database=%s", self._db_name)
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            logger.critical("MongoDB connection failed: %s", e, exc_info=True)
            raise

    @property
    def drivers(self):
        return self._db["drivers"]

    @property
    def sessions(self):
        return self._db["sessions"]

    @property
    def alerts(self):
        return self._db["alerts"]

    @property
    def daily_scores(self):
        return self._db["daily_scores"]


_client = MongoDBClient()
drivers_collection = _client.drivers
sessions_collection = _client.sessions
alerts_collection = _client.alerts
daily_scores_collection = _client.daily_scores
