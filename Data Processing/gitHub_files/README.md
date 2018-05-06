# local-weather
Utilizes station location metadata and NOAA Quality Controlled Local Climatological Data (QCLCD) to provide up to date location-specific weather data

It automatically downloads weather data for a given time period and then finds the relevant data for a specific set of stations near a zip code. Try running `python TestWeatherData.py` to make sure the basic functionality works.

Then try this to dump data for several zip codes and time ranges (defined in `test\example_config.csv`) to a unified csv file, using the 5 closest stations to each zip code, but limited to a 10 km search radius.

`python weatherDump.py -i test\example_config.csv -o example_dump.csv -n 5 -d 10`
