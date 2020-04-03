import calendar
import datetime
import os
from itertools import islice, tee
from typing import Iterable, List, Optional
from warnings import warn

import numpy as np
import pandas as pd
from click import confirm
from tabulate import tabulate

from ipl import image_analysis as image_anal
from ipl import importexport as importexport
from ipl import visualization as visualization
from ipl.database import image_db as image_db
from ipl.errors import IPLError
from ipl.logging_ import logger


def _add_months(initial_date: datetime.date,
                months: int):
    month = initial_date.month - 1 + months
    year = initial_date.year + month // 12
    month = month % 12 + 1
    month_range = calendar.monthrange(year, month)[1]
    day = min(initial_date.day, month_range)
    return datetime.date(year, month, day)


def _collect_cloudiness_data(cloudiness_data: Iterable[float],
                             epsilon: float = 0.01):
    """Returns non-clouded, partially clouded and fully clouded count"""

    def generate_estimation_tuple(cloudiness):
        return (cloudiness < epsilon,
                epsilon < cloudiness < 1 - epsilon,
                cloudiness > 1 - epsilon)

    generate_tuples = (generate_estimation_tuple(cloudiness) for cloudiness in cloudiness_data)
    non_gen, partially_gen, fully_gen = tee(generate_tuples, 3)
    return (sum(1 for non, partially, fully in non_gen if non),
            sum(1 for non, partially, fully in partially_gen if partially),
            sum(1 for non, partially, fully in fully_gen if fully))


def _calculate_images_statistics(images_array: List[np.ndarray],
                                 index_column: Optional[np.ndarray] = None):
    cloud_rate = np.empty(len(images_array), np.double)
    average = np.empty(len(images_array), np.double)
    std = np.empty(len(images_array), np.double)
    ci_lower = np.empty(len(images_array), np.double)
    ci_upper = np.empty(len(images_array), np.double)

    for index_, image in enumerate(images_array):
        average[index_] = np.mean(image)
        std[index_] = np.std(image)
        ci_lower[index_], ci_upper[index_] = image_anal.calculate_confidence_interval(image)
        cloud_rate[index_] = image_anal.calculate_clouds_percentile(image)

    data_sources = (cloud_rate, average, std, ci_lower, ci_upper)
    columns = ['cloud_rate', 'ndvi_average', 'standard_deviation', 'lower_ci', 'upper_ci']
    df_initializer = {label: source for label, source in zip(columns, data_sources)}
    dataframe = pd.DataFrame(df_initializer)
    if index_column is not None:
        dataframe['Image ID'] = index_column
        dataframe.set_index('Image ID')
    return dataframe


def _collect_images(field_ids: List[int],
                    start_date: datetime.date,
                    end_date: datetime.date,
                    filtered_columns: Optional[List[str]] = None,
                    should_sort_images: bool = True,
                    limit: Optional[int] = None):
    database = image_db.ImageDatabaseInstance()

    select_images = database.select_field_images
    dataframes_list: List[pd.DataFrame] = []
    for field_id in field_ids:
        data_limit = limit - len(dataframes_list) if limit is not None else None
        field_images = select_images(field_id=field_id,
                                     date_start=start_date,
                                     date_end=end_date,
                                     filtered_columns=filtered_columns,
                                     limit=data_limit)
        dataframes_list.append(field_images)
        if limit is not None and len(dataframes_list) >= limit:
            break

    if dataframes_list:
        return pd.concat(dataframes_list, sort=should_sort_images)
    else:
        return None


def require_extension_modules(dependencies_list):
    import importlib.util as import_utils
    for package in dependencies_list:
        spam_spec = import_utils.find_spec(package)
        found = spam_spec is not None
        if not found:
            warn(f'Unable to found {package} in installed dependencies, some operations may not run successfully',
                 category=RuntimeWarning)


def process_images(file: Optional[str],
                   image_ids: Optional[List[int]],
                   export_location: str,
                   all_: bool,
                   cache: bool,
                   **kwargs):
    if file:
        processed_images = [importexport.read_image_bitmap(file)]
    else:
        database = image_db.ImageDatabaseInstance()
        if all_:
            image_ids = database.select_images_ids()
        required_fields = ['image_data']
        images_data = [database.select_image(image_id, required_fields) for image_id in image_ids]
        processed_images = [data['image_data'][0] for data in images_data]

    image_ids = np.array(image_ids, dtype=image_anal.IMAGE_DATA_TYPE)
    dataframe = _calculate_images_statistics(processed_images, image_ids)
    labels = [label.replace('_', ' ').capitalize() for label in dataframe.columns.values]
    print(tabulate(dataframe, headers=labels, tablefmt='psql'))
    if cache and image_ids:
        database = image_db.ImageDatabaseInstance()
        for image_id, (index, row) in zip(image_ids, dataframe.iterrows()):
            if not database.check_if_has_cached_statistics(image_id):
                cloud_rate = row['cloud_rate']
                ndvi_average = row['ndvi_average']
                std = row['standard_deviation']
                lower_ci = row['lower_ci']
                upper_ci = row['upper_ci']
                database.insert_image_statistics(image_id, cloud_rate, ndvi_average,
                                                 std, lower_ci, upper_ci)
    if export_location:
        require_extension_modules(['xlsxwriter'])
        dataframe.to_excel(export_location, engine='xlsxwriter',
                           na_rep='N/A', sheet_name='Statistics')


def database_view(field_ids: Optional[List[int]],
                  head: Optional[int],
                  start: datetime.date,
                  end: datetime.date,
                  **kwargs):
    required_columns = ['field_id', 'image_id', 'revision',
                        'capture_date', 'mysterious_date', 'capture_satellite']
    database = image_db.ImageDatabaseInstance()
    if not field_ids:
        field_ids = database.select_fields_ids()
    dataframe = _collect_images(field_ids=field_ids,
                                start_date=start,
                                end_date=end,
                                filtered_columns=required_columns,
                                limit=head)
    if dataframe is not None:
        shown_dataframe = dataframe.head(head) if head is not None else dataframe
        shown_dataframe = shown_dataframe.sort_values(by=['field_id', 'image_id', 'revision'])
        labels = [label.replace('_', ' ').capitalize() for label in required_columns]
        print(tabulate(shown_dataframe.loc[:, required_columns], headers=labels, tablefmt='psql', showindex=False))
    else:
        print("No suitable records found !")


def visualize_clouds(field_id: int,
                     start: datetime.date,
                     end: datetime.date,
                     **kwargs):
    database = image_db.ImageDatabaseInstance()
    required_columns = ['image_id', 'cloud_rate', 'capture_date']
    cached_statistics = database.select_field_statistics(field_id,
                                                         filtered_columns=required_columns,
                                                         date_start=start,
                                                         date_end=end)
    calculated_images_set = set(cached_statistics['image_id'])
    cloud_rates = list(cached_statistics['cloud_rate'])
    capture_dates = list(cached_statistics['capture_date'])
    del cached_statistics

    required_columns = ['image_id', 'image_data', 'capture_date']
    all_images = database.select_field_images(field_id=field_id,
                                              filtered_columns=required_columns,
                                              date_start=start,
                                              date_end=end)
    for index, row in all_images.iterrows():
        image_id = row['image_id']
        if image_id not in calculated_images_set:
            image_bitmap = row['image_data']
            cloud_rate = image_anal.calculate_clouds_percentile(image_bitmap)
            cloud_rates.append(cloud_rate)
            capture_dates.append(row['capture_date'])
    assert len(cloud_rates) == len(capture_dates)

    if len(capture_dates) > 0:
        capture_dates, cloud_rates = zip(*sorted(zip(capture_dates, cloud_rates)))
        minimal_goal_date = _add_months(capture_dates[0], 1)
        dates_list = []
        statistics_arrays = [[], [], []]
        index_start, index_end = None, 0

        def _update_statistics(new_date: datetime.date):
            dates_list.append(new_date)
            cloud_rates_generator = islice(cloud_rates, index_start, index_end)
            for index_, statistics_item in enumerate(_collect_cloudiness_data(cloud_rates_generator)):
                statistics_arrays[index_].append(statistics_item)

        for index, date in enumerate(capture_dates):
            if date > minimal_goal_date:
                index_start, index_end = index_end, index
                _update_statistics(date)
                minimal_goal_date = _add_months(date, 1)
        index_start, index_end = index_end, len(capture_dates)
        _update_statistics(minimal_goal_date)
        visualization.plot_clouds_impact_for_a_period(dates_list, *statistics_arrays)
        visualization.show_plots()
    else:
        print('Unable to start visualization, no data found')


def visualize_occurrences(file: Optional[str],
                          image_id: Optional[int],
                          **kwargs):
    if file:
        bitmap = importexport.read_image_bitmap(file)
    elif image_id is not None:
        database = image_db.ImageDatabaseInstance()
        bitmap = database.select_image(image_id)['image_data'][0]
    else:
        raise IPLError('Specify file or image_id for occurrences visualization')

    unique_value_occurs = image_anal.construct_values_occurrences_map(bitmap)
    if 0 in unique_value_occurs.keys():
        del unique_value_occurs[0]
    visualization.plot_values_frequencies(unique_value_occurs)
    visualization.show_plots()


def visualize_statistics(field_ids: List[str],
                         start: datetime.date,
                         end: datetime.date,
                         max_cloudiness: float,
                         **kwargs):
    database = image_db.ImageDatabaseInstance()
    required_fields = ['image_id', 'capture_date', 'index_weighted_avg',
                       'confidence_interval_lower', 'confidence_interval_upper']
    have_plotted_anything = False
    for field_id in field_ids:
        cached_statistics = database.select_field_statistics(field_id=field_id,
                                                             filtered_columns=required_fields,
                                                             date_start=start,
                                                             date_end=end,
                                                             max_cloudiness=max_cloudiness)
        print(cached_statistics)
        cached_images_ids = set(cached_statistics['image_id'])
        locally_required_fields = ['image_id', 'image_data', 'capture_date']
        other_images = database.select_field_images(field_id=field_id,
                                                    filtered_columns=locally_required_fields,
                                                    date_start=start,
                                                    date_end=end)
        for index, image in other_images.iterrows():
            image_id = image['image_id']
            if image_id not in cached_images_ids:
                bitmap: np.ndarray = image['image_data']
                cloud_rate = image_anal.calculate_clouds_percentile(bitmap)
                if cloud_rate <= max_cloudiness:
                    capture_date = image['capture_date']
                    mean = np.nanmean(bitmap)
                    lower_ci, upper_ci = image_anal.calculate_confidence_interval(bitmap)
                    series = pd.Series([image_id, capture_date, mean, lower_ci, upper_ci],
                                       index=cached_statistics.columns)
                    cached_statistics = cached_statistics.append(series, ignore_index=True)

        if cached_statistics.shape and cached_statistics.shape[0]:
            cached_statistics.sort_values(by='capture_date', inplace=True)
            visualization.plot_statistics_for_a_period(time_stamps=cached_statistics['capture_date'],
                                                       mean=cached_statistics['index_weighted_avg'],
                                                       lower_ci=cached_statistics['confidence_interval_lower'],
                                                       upper_ci=cached_statistics['confidence_interval_upper'],
                                                       legend_name=str(field_id))
            have_plotted_anything = True

    if have_plotted_anything:
        visualization.show_plots()
    else:
        print('Unable to start visualization, no data found')


def import_images(import_location: List[str],
                  cache: bool,
                  **kwargs):
    database = image_db.ImageDatabaseInstance()
    for location in import_location:
        if os.path.isdir(location):
            images_data = importexport.import_images_folder(location)
        else:
            images_data = filter(lambda data: data is not None,
                                 (importexport.import_locally_stored_image(location),))
        for image_data in images_data:
            file_path, db_info = image_data
            image_id = database.insert_image(*db_info)
            if image_id is not None:
                if cache:
                    bitmap = db_info[2]
                    statistics = image_anal.calculate_all_statistics(bitmap)
                    database.insert_image_statistics(image_id, *statistics)
            else:
                logger.warning('Unable to import file "%s", it already exists in a database', file_path)


def export_images(export_location: str,
                  start: datetime.date,
                  end: datetime.date,
                  driver: str,
                  field_ids: List[int],
                  all_: bool,
                  force: bool,
                  **kwargs):
    if not os.path.isdir(export_location):
        if force:
            os.makedirs(export_location, exist_ok=True)
        else:
            raise IPLError('Unable to export data to non-existent directory')
    database = image_db.ImageDatabaseInstance()
    if all_:
        field_ids = database.select_fields_ids()
    dataframe = _collect_images(field_ids=field_ids,
                                start_date=start,
                                end_date=end)
    selected_extension = importexport.SupportedDrivers[driver].value

    for index, row in dataframe.iterrows():
        capture_date = row['capture_date'].strftime("%d%m%Y")
        satellite = row['capture_satellite']
        mysterious_date = row['mysterious_date'].strftime("P%Y%m%d")
        field_id = row['field_id']
        revision = row['revision']
        bitmap = row['image_data']
        file_name = (f'{capture_date}_{field_id}r{revision}_NDVI_'
                     f'{mysterious_date}_{satellite}.{selected_extension}')
        file_path = os.path.join(export_location, file_name)
        importexport.write_image_bitmap(file_path, bitmap, driver)


def reset_database(confirmed: bool,
                   **kwargs):
    if not confirmed:
        confirmed = confirm('Do you really want to erase all stored data ?')
    if confirmed:
        database = image_db.ImageDatabaseInstance()
        database.erase_all()
