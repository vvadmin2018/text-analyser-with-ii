from src.feature_extractor import FeatureExtractor, Language
from src.profile_builder import AuthorProfile


class FuzzyDetective:
    """Нечёткий детектив - главный класс программы"""

    def __init__(self, language=Language.RUSSIAN):
        self.authors = {}  # словарь {имя: AuthorProfile}
        self.language = language

    def add_author(self, name, texts):
        """Добавляет автора с его текстами для обучения"""
        profile = AuthorProfile(name)
        profile.build_from_texts(texts, language=self.language)
        self.authors[name] = profile

    def identify(self, anonymous_text, threshold=0.5):
        """
        Определяет автора анонимного текста

        Returns:
            tuple: (best_author, results, similarity_details)
            results — {имя: итоговое сходство}
            similarity_details — {имя: (similarities, weights, contributions)},
            тот же формат, что и AuthorProfile.similarity_with_details(),
            удобно передавать напрямую в StyleRose.plot_membership_rose /
            StyleRose.plot_feature_importance.
        """
        extractor = FeatureExtractor(language=self.language)
        features = extractor.extract(anonymous_text)

        results = {}
        similarity_details = {}
        for name, profile in self.authors.items():
            sim, details = profile.similarity_with_details(features)
            results[name] = sim
            similarity_details[name] = details

        # Находим лучшего
        best_author = max(results, key=results.get)
        best_score = results[best_author]

        if best_score < threshold:
            return "Автор не определён", results, similarity_details
        else:
            return best_author, results, similarity_details
