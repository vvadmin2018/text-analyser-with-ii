# src/visualization/visualizer.py
"""Визуализация результатов стилометрического анализа"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from matplotlib.backends.backend_pdf import PdfPages

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class StyleRose:
    """Построение роз ветров для визуализации стилевых профилей"""
    
    @staticmethod
    def normalize_by_max(authors_profiles: Dict[str, List[float]], 
                         anonymous_vector: np.ndarray) -> Tuple[Dict, List]:
        """Нормализует признаки делением на максимум"""
        n_features = len(anonymous_vector)
        
        # Собираем все значения
        feature_values = [[] for _ in range(n_features)]
        for values in authors_profiles.values():
            for i, val in enumerate(values):
                feature_values[i].append(val)
        for i, val in enumerate(anonymous_vector):
            feature_values[i].append(val)
        
        # Нормализуем
        normalized_profiles = {}
        for name, values in authors_profiles.items():
            norm_values = []
            for i, val in enumerate(values):
                max_val = max(feature_values[i])
                norm_values.append(val / max_val if max_val > 0 else 0)
            normalized_profiles[name] = norm_values
        
        normalized_anon = []
        for i, val in enumerate(anonymous_vector):
            max_val = max(feature_values[i])
            normalized_anon.append(val / max_val if max_val > 0 else 0)
        
        return normalized_profiles, normalized_anon
    
    @staticmethod
    def plot_fuzzy_rose(authors_profiles: Dict[str, List[float]],
                        anonymous_vector: np.ndarray,
                        feature_names: List[str],
                        profiles_dispersion: Optional[Dict[str, List[float]]] = None,
                        anonymous_dispersion: Optional[List[float]] = None,
                        title: str = "Нечёткая роза ветров",
                        author_colors: Optional[Dict[str, str]] = None,
                        figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Строит нечёткую розу ветров с размытыми границами
        
        Args:
            authors_profiles: профили авторов {имя: значения}
            anonymous_vector: вектор анонимного текста
            feature_names: названия признаков
            profiles_dispersion: дисперсия профилей
            anonymous_dispersion: дисперсия анонима
            title: заголовок графика
            author_colors: цвета для авторов
            figsize: размер фигуры
            
        Returns:
            matplotlib фигура
        """
        n_features = len(feature_names)
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]  # Замыкаем круг
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
        
        # Цвета по умолчанию
        default_colors = ['#FF4B4B', '#4B7BFF', '#FFB44B', '#4BFF4B', '#B44BFF']
        
        # Рисуем каждого автора
        for idx, (name, values) in enumerate(authors_profiles.items()):
            color = author_colors.get(name, default_colors[idx % len(default_colors)]) if author_colors else default_colors[idx % len(default_colors)]
            
            values_closed = values + [values[0]]
            
            # Основная линия
            ax.plot(angles, values_closed, 'o-', linewidth=2.5, color=color, 
                   label=name, markersize=8, alpha=0.8)
            
            # Область размытия (если есть дисперсия)
            if profiles_dispersion and name in profiles_dispersion:
                dispersion = profiles_dispersion[name]
                upper = [min(1.0, v + d) for v, d in zip(values, dispersion)]
                lower = [max(0.0, v - d) for v, d in zip(values, dispersion)]
                
                upper_closed = upper + [upper[0]]
                lower_closed = lower + [lower[0]]
                
                ax.fill(angles, upper_closed, color=color, alpha=0.1)
                ax.fill(angles, lower_closed, color=color, alpha=0.1)
            
            # Заполненная область
            ax.fill(angles, values_closed, color=color, alpha=0.15)
        
        # Анонимный текст
        anon_color = '#2ECC71'  # зелёный
        anon_values = anonymous_vector.tolist() if not isinstance(anonymous_vector, list) else anonymous_vector
        anon_closed = anon_values + [anon_values[0]]
        
        ax.plot(angles, anon_closed, 's-', linewidth=3, color=anon_color,
               label='Аноним', markersize=10, alpha=0.9)
        
        # Дисперсия анонима
        if anonymous_dispersion:
            upper = [min(1.0, v + d) for v, d in zip(anon_values, anonymous_dispersion)]
            lower = [max(0.0, v - d) for v, d in zip(anon_values, anonymous_dispersion)]
            upper_closed = upper + [upper[0]]
            lower_closed = lower + [lower[0]]
            
            ax.fill(angles, upper_closed, color=anon_color, alpha=0.1)
            ax.fill(angles, lower_closed, color=anon_color, alpha=0.1)
        
        # Настройки осей
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # Сетка
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_feature_importance(author_name: str,
                                similarities: List[float],
                                weights: List[float],
                                contributions: List[float],
                                feature_names: List[str],
                                title: str = "Важность признаков") -> plt.Figure:
        """
        Строит график важности признаков
        
        Returns:
            matplotlib фигура
        """
        n_features = len(feature_names)
        x = np.arange(n_features)
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - width, similarities, width, label='Сходство (μ)', color='#4B7BFF')
        bars2 = ax.bar(x, weights, width, label='Вес', color='#FFB44B')
        bars3 = ax.bar(x + width, contributions, width, label='Вклад', color='#4BFF4B')
        
        ax.set_xlabel('Признак')
        ax.set_ylabel('Значение')
        ax.set_title(f'{title}\n{author_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_authors_comparison(results: Dict[str, float],
                                title: str = "Сравнение уверенности идентификации") -> plt.Figure:
        """
        Строит сравнительную диаграмму авторов
        
        Returns:
            matplotlib фигура
        """
        authors = list(results.keys())
        scores = list(results.values())
        
        colors = ['#FF4B4B', '#4B7BFF', '#FFB44B', '#4BFF4B', '#B44BFF'][:len(authors)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(authors, scores, color=colors, alpha=0.8)
        
        # Добавляем значения на столбцы
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.1%}',
                   ha='center', va='bottom', fontsize=11)
        
        ax.set_ylabel('Уверенность')
        ax.set_title(title)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Порог 50%')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_feature_heatmap(authors_data: Dict[str, List[float]],
                             feature_names: List[str],
                             title: str = "Тепловая карта признаков") -> plt.Figure:
        """
        Строит тепловую карту признаков
        
        Returns:
            matplotlib фигура
        """
        authors = list(authors_data.keys())
        data = np.array([authors_data[name] for name in authors])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        
        # Подписи
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(authors)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_yticklabels(authors)
        
        # Значения в ячейках
        for i in range(len(authors)):
            for j in range(len(feature_names)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label='Нормализованное значение')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_pdf_report(profiles: Dict,
                          anon_features: np.ndarray,
                          results: Dict,
                          similarity_details: Dict,
                          anonymous_file: str,
                          best_author: str,
                          output_path: str = 'style_analysis_report.pdf'):
        """
        Создаёт PDF отчёт со всеми графиками
        """
        feature_names = [
            'Дл. Предл', 'Дисп.', 'Абз', '?', '!', '...', 'Прям. речь',
            'TTR', 'Сущ', 'Глаг', 'Прил', 'ДлСл', ',', '—', ':', 'Союз', 'Предлог'
        ]
        
        # Подготавливаем данные для визуализации
        authors_for_plot = {}
        authors_dispersion = {}
        
        for name, profile in profiles.items():
            typical_values = [f.b for f in profile.features]
            dispersion = [(f.c - f.a) / 4 for f in profile.features]
            authors_for_plot[name] = typical_values
            authors_dispersion[name] = dispersion
        
        anon_dispersion = []
        for i in range(len(anon_features)):
            avg_disp = np.mean([authors_dispersion.get(name, [0]*17)[i] 
                               for name in authors_dispersion])
            anon_dispersion.append(avg_disp)
        
        with PdfPages(output_path) as pdf:
            # Титульная страница
            fig = plt.figure(figsize=(11, 8.5))
            plt.axis('off')
            plt.text(0.5, 0.7, 'Анализ авторства текста',
                     fontsize=24, ha='center', fontweight='bold')
            plt.text(0.5, 0.6, 'Нечёткая логика и стилевые профили',
                     fontsize=18, ha='center')
            plt.text(0.5, 0.4, f'Анонимный текст: {anonymous_file}',
                     fontsize=14, ha='center')
            plt.text(0.5, 0.3, f'Определён автор: {best_author} (уверенность {results[best_author]:.1%})',
                     fontsize=14, ha='center', color='green')
            pdf.savefig()
            plt.close(fig)
            
            # Индивидуальные розы
            for name in authors_for_plot:
                single_dict = {name: authors_for_plot[name]}
                single_disp = {name: authors_dispersion[name]}
                
                fig = StyleRose.plot_fuzzy_rose(
                    single_dict, anon_features, feature_names,
                    profiles_dispersion=single_disp,
                    anonymous_dispersion=anon_dispersion,
                    title=f"Сравнение: {name} vs аноним"
                )
                pdf.savefig()
                plt.close(fig)
            
            # Общая роза
            fig = StyleRose.plot_fuzzy_rose(
                authors_for_plot, anon_features, feature_names,
                profiles_dispersion=authors_dispersion,
                anonymous_dispersion=anon_dispersion,
                title="Нечёткая роза ветров: все авторы"
            )
            pdf.savefig()
            plt.close(fig)
            
            # Важность признаков
            for name, (sims, weights, contribs) in similarity_details.items():
                fig = StyleRose.plot_feature_importance(
                    name, sims, weights, contribs, feature_names,
                    title=f"Важность признаков: {name}"
                )
                pdf.savefig()
                plt.close(fig)
            
            # Сравнение авторов
            fig = StyleRose.plot_authors_comparison(results)
            pdf.savefig()
            plt.close(fig)
            
            # Тепловая карта
            fig = StyleRose.plot_feature_heatmap(authors_for_plot, feature_names)
            pdf.savefig()
            plt.close(fig)
        
        print(f"✅ PDF отчёт сохранён: {output_path}")


class InteractiveVisualizer:
    """Интерактивная визуализация с использованием Plotly"""
    
    @staticmethod
    def create_interactive_rose(authors_profiles: Dict[str, List[float]],
                                anonymous_vector: np.ndarray,
                                feature_names: List[str],
                                title: str = "Интерактивная роза ветров"):
        """
        Создаёт интерактивную розу ветров с Plotly
        
        Returns:
            Plotly figure
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️  Plotly не установлен. Используем matplotlib.")
            return None
        
        n_features = len(feature_names)
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]
        
        fig = make_subplots(specs=[[{"type": "polar"}]])
        
        colors = ['#FF4B4B', '#4B7BFF', '#FFB44B', '#4BFF4B', '#B44BFF']
        
        # Авторы
        for idx, (name, values) in enumerate(authors_profiles.items()):
            values_closed = values + [values[0]]
            color = colors[idx % len(colors)]
            
            fig.add_trace(go.Barp(
                r=values_closed,
                theta=angles,
                name=name,
                marker_color=color,
                opacity=0.6,
                line_color=color,
                line_width=2
            ))
        
        # Аноним
        anon_values = anonymous_vector.tolist() if not isinstance(anonymous_vector, list) else anonymous_vector
        anon_closed = anon_values + [anon_values[0]]
        
        fig.add_trace(go.Barp(
            r=anon_closed,
            theta=angles,
            name='Аноним',
            marker_color='#2ECC71',
            opacity=0.8,
            line_color='#2ECC71',
            line_width=3
        ))
        
        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                angularaxis=dict(direction="clockwise")
            ),
            showlegend=True,
            height=600
        )
        
        return fig
    
    @staticmethod
    def save_html(fig, filename: str = 'interactive_report.html'):
        """Сохраняет интерактивный отчёт в HTML"""
        if not PLOTLY_AVAILABLE:
            return
        
        fig.write_html(filename)
        print(f"✅ Интерактивный отчёт сохранён: {filename}")
