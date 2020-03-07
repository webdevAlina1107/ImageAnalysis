from datetime import datetime
from typing import Any, Dict, Optional, Sequence

import matplotlib.dates as dates
import matplotlib.pyplot as plot
import numpy as np

from ipl.image_analysis import IMAGE_DATA_TYPE


def _construct_bar(ax,
                   y_values: Sequence[int],
                   x_values: Optional[Sequence[Any]] = None,
                   bar_width: float = 0.9,
                   should_label_bars: bool = True,
                   **kwargs):
    if not x_values:
        x_values = range(len(y_values))

    rectangles = ax.bar(x_values,
                        y_values,
                        align='center',
                        width=bar_width,
                        **kwargs)

    if should_label_bars:
        _label_hist_rectangles(rectangles)

    return rectangles


def _setup_axes_for_hist(ax,
                         title: str,
                         x_label: str,
                         y_label: str):
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')


def _label_hist_rectangles(rectangles,
                           text_offset: float = 0.3):
    for rect in rectangles:
        height = rect.get_height()
        if height > 0:
            plot.text(rect.get_x() + rect.get_width() / 2.,
                      height + text_offset,
                      f'{height}',
                      ha='center',
                      va='bottom')


def visualize_values_frequencies(unique_values_occurrences: Dict[IMAGE_DATA_TYPE, int],
                                 **kwargs):
    figure, axes = plot.subplot(1, 1, 1)
    occurrences = np.fromiter(unique_values_occurrences.values(), dtype=IMAGE_DATA_TYPE)
    _setup_axes_for_hist(axes,
                         title='Frequencies histogram',
                         x_label='Values',
                         y_label='Occurrences')
    _construct_bar(axes, occurrences, **kwargs)
    values_ = np.fromiter(unique_values_occurrences.keys(), dtype=IMAGE_DATA_TYPE)
    axes.set_xticks(range(len(values_)))
    axes.set_xticklabels(values_)
    plot.show()


def visualize_clouds_impact(date_stamps: Sequence[datetime],
                            non_clouded_counts: Sequence[int],
                            partially_clouded_counts: Sequence[int],
                            fully_clouded_counts: Sequence[int]):
    """Accepts date stamps array with cloud statistics for each date
    and plots bars histogram with this data"""
    figure, axes = plot.subplot(1, 1, 2)
    _setup_axes_for_hist(axes,
                         title='Clouded images statistics',
                         x_label='Time stamps',
                         y_label='Count of images')
    dates_formatter = dates.DateFormatter('%d/%m/%Y')
    dates_locator = dates.MonthLocator()
    axes.xaxis.set_major_locator(dates_locator)
    axes.xaxis.set_major_formatter(dates_formatter)

    colors = ['r', 'g', 'b']
    bars_height = [non_clouded_counts, partially_clouded_counts, fully_clouded_counts]

    def create_bar(y_axis, color):
        return _construct_bar(axes,
                              y_values=y_axis,
                              x_values=date_stamps,
                              bar_width=0.3,
                              color=color)

    bars = tuple(create_bar(data, color) for data, color in zip(bars_height, colors))
    labels = ['Non clouded', 'Partially clouded', 'Fully clouded']
    axes.legend(bars, labels)
    axes.xaxis_date()
    figure.autofmt_xdate()
