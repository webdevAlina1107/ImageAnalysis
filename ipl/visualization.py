from datetime import datetime
from typing import Any, Dict, Optional, Sequence

import matplotlib.dates as dates
import matplotlib.pyplot as plot
import numpy as np

from ipl.image_analysis import IMAGE_DATA_TYPE
from ipl._logging import logger


def _construct_bar(ax,
                   y_values: Sequence[int],
                   x_values: Optional[Sequence[Any]] = None,
                   bar_width: float = 0.9,
                   should_label_bars: bool = True,
                   **kwargs):
    if x_values is None:
        x_values = range(len(y_values))

    logger.debug(f'Plotting bar with width {bar_width}')
    rectangles = ax.bar(x_values,
                        y_values,
                        align='center',
                        width=bar_width,
                        **kwargs)

    if should_label_bars:
        _label_hist_rectangles(rectangles)

    return rectangles


def _setup_axes(ax,
                title: str,
                x_label: str,
                y_label: str):
    logger.debug('Setting labels')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    logger.debug('Configuring grid')
    ax.grid(True, axis='y')
    logger.debug('Making plot looking more prettier')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')


def _label_hist_rectangles(rectangles,
                           text_offset: float = 0.3):
    logger.debug(f'Labeling histogram rectangles with text offset = {text_offset}')
    for rect in rectangles:
        height = rect.get_height()
        if height > 0:
            plot.text(rect.get_x() + rect.get_width() / 2.,
                      height + text_offset,
                      f'{height}',
                      ha='center',
                      va='bottom')


def _determine_date_locator(dates_range: Sequence[datetime]):
    min_date, max_date = min(dates_range), max(dates_range)
    diff = max_date - min_date
    logger.debug(f'Selecting date locator for timeline diff = {diff}')
    locator = dates.YearLocator()
    locator_label = 'year'
    if diff.days < 31:
        locator_label = 'days'
        locator = dates.DayLocator()
    if diff.days < 365:
        locator_label = 'month'
        locator = dates.MonthLocator()

    logger.debug(f'Selected {locator_label} date locator')
    return locator


def plot_values_frequencies(unique_values_occurrences: Dict[IMAGE_DATA_TYPE, int],
                            **kwargs):
    axes, figure = plot.gca(), plot.gcf()
    occurrences = np.fromiter(unique_values_occurrences.values(), dtype=IMAGE_DATA_TYPE)
    logger.debug('Configuring axes')
    _setup_axes(axes,
                title='Frequencies histogram',
                x_label='Values',
                y_label='Occurrences')
    logger.debug('Constructing occurrences bar')
    _construct_bar(axes, occurrences, **kwargs)
    values_ = np.fromiter(unique_values_occurrences.keys(), dtype=IMAGE_DATA_TYPE)
    logger.debug('Configuring X-labels')
    axes.set_xticks(range(len(values_)))
    axes.set_xticklabels(values_)


def plot_clouds_impact_for_a_period(time_stamps: Sequence[datetime],
                                    non_clouded_counts: Sequence[int],
                                    partially_clouded_counts: Sequence[int],
                                    fully_clouded_counts: Sequence[int]):
    """Accepts date stamps array with cloud statistics for each date
    and plots bars histogram with this data"""
    axes, figure = plot.gca(), plot.gcf()
    logger.debug('Configuring histogram axes')
    _setup_axes(axes,
                title='Clouded images statistics',
                x_label='Time stamps',
                y_label='Count of images')

    logger.debug('Configuring date formatting')
    dates_formatter = dates.DateFormatter('%d/%m/%Y')
    dates_locator = _determine_date_locator(time_stamps)
    axes.xaxis.set_major_locator(dates_locator)
    axes.xaxis.set_major_formatter(dates_formatter)

    colors = ['mediumslateblue',
              'plum',
              'thistle']
    bars_height = [non_clouded_counts, partially_clouded_counts, fully_clouded_counts]
    time_stamps = dates.date2num(time_stamps)
    bar_width = 0.2

    def create_bar(y_axis, color, n=0):
        # Asserting that there would be only 3 bars
        return _construct_bar(axes,
                              y_values=y_axis,
                              x_values=time_stamps + bar_width * n - bar_width,
                              bar_width=bar_width,
                              color=color)

    logger.debug('Creating 3 bars')
    bars = tuple(create_bar(data, color, index) for index, (data, color) in enumerate(zip(bars_height, colors)))
    logger.debug('Update legend')
    labels = ['Non clouded', 'Partially clouded', 'Fully clouded']
    axes.legend(bars, labels)

    logger.debug('Formatting figure')
    axes.xaxis_date()
    axes.autoscale(tight=True)
    figure.autofmt_xdate()


def plot_statistics_for_a_period(time_stamps: Sequence[datetime],
                                 mean: Sequence[float],
                                 lower_ci: Sequence[float],
                                 upper_ci: Sequence[float],
                                 legend_name: Optional[str] = None):
    axes, figure = plot.gca(), plot.gcf()
    logger.debug('Configuring axes')
    _setup_axes(axes,
                title='Time period CI statistics',
                x_label='Time stamps',
                y_label='Statistical data')
    logger.debug('Configuring date formatting')
    dates_formatter = dates.DateFormatter('%d/%m/%Y')
    dates_locator = _determine_date_locator(time_stamps)
    axes.xaxis.set_major_locator(dates_locator)
    axes.xaxis.set_major_formatter(dates_formatter)

    logger.debug('Selecting color')
    random_line_color = np.random.rand(3)
    logger.debug('Plotting CI')
    plot.fill_between(time_stamps, lower_ci, upper_ci,
                      color=random_line_color,
                      alpha=0.5)

    logger.debug('Plotting mean line')
    if legend_name:
        plot.plot(time_stamps, mean,
                  color=random_line_color,
                  label=legend_name)
        axes.legend()
    else:
        plot.plot(time_stamps, mean,
                  color=random_line_color)

    logger.debug('Formatting figure')
    axes.xaxis_date()
    axes.autoscale(tight=True)
    figure.autofmt_xdate()


def show_plots():
    logger.debug('Showing plots')
    plot.show()
