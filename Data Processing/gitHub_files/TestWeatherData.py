import unittest
import time
import datetime
import WeatherData as weather
from pandas.testing import assert_frame_equal  # Updated import path

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(f'[{self.name}]', end=' ')
        print(f'Elapsed: {time.time() - self.tstart}')

class WeatherTest(unittest.TestCase):

    def setUp(self):
        self.wd = weather.WeatherData('weather')  # dataDir

    def test_weatherRange(self):  # Fixed method name
        zip5 = 12601
        zips = self.wd.zipMap()
        print(zips[zip5])
        with Timer('weather'):
            start = datetime.datetime(2013, 3, 1)
            end = datetime.datetime(2013, 4, 21)
            dt = datetime.timedelta(days=1)
            dates = [start + x * dt for x in range((end - start).days)]
            start = datetime.datetime(2013, 6, 1)
            end = datetime.datetime(2013, 6, 5)
            weather = self.wd.weatherRange(zip5, start, end, True)
            print(weather)

    def test_stationList(self):
        closest5 = self.wd.stationList(12601, y=2013, m=3, n=5)
        self.assertTrue(len(closest5) == 5)
        print(f"Top {len(closest5)} stations closest to {12601}:")
        print("  WBAN, dist (km), name")
        for sta in closest5:
            print(f"  {self.wd.summarizeStation(sta)}")
        print('')

        within10 = self.wd.stationList(12601, y=2013, m=3, n=0, preferredDistKm=10)
        self.assertTrue(len(within10) == 1)
        print(f"{len(within10)} station(s) within {10} km from {12601}:")
        print("  WBAN, dist (km), name")
        for sta in within10:
            print(f"  {self.wd.summarizeStation(sta)}")
        print('')

        within30 = self.wd.stationList(94568, y=2013, m=3, n=0, preferredDistKm=30)
        print(f"{len(within30)} station(s) within {30} km from {94568}.")
        print("  WBAN, dist (km), name")
        for sta in within30:
            print(f"  {self.wd.summarizeStation(sta)}")
        self.assertTrue(len(within30) == 3)
        print('')

        broken = self.wd.stationList(95223, y=2013, m=3, n=3)
        print(f"{len(broken)} station(s) within {30} km from {95223}.")
        print("  WBAN, dist (km), name")
        for sta in broken:
            print(f"  {self.wd.summarizeStation(sta)}")

    def test_weatherMonth(self):
        with self.assertRaises(KeyError):
            self.wd.weatherMonth(999, 2013, 3, hourly=False)
        marchDataN = self.wd.weatherMonth(12601, 2013, 3, hourly=False, n=5)
        self.assertEqual(len(marchDataN), 124)

        marchDataDist = self.wd.weatherMonth(12601, 2013, 3, hourly=False, n=0, preferredDistKm=40)
        self.assertEqual(len(marchDataDist), 93)

        marchDataDist = self.wd.weatherMonth([12601, 94611], 2013, 3, hourly=False, n=0, preferredDistKm=40)
        self.assertEqual(len(marchDataDist), 279)

        flat1 = self.wd.combineStacks(self.wd.stackDailyWeatherData(marchDataN), addValues=[('zip5', 12601)])
        flat2 = self.wd.combineStacks(self.wd.weatherMonth(12601, 2013, 3, hourly=False, n=5, stackData=True), addValues=[('zip5', 12601)])
        assert_frame_equal(flat1, flat2)

        eastStations = [s[0] for s in self.wd.stationList(12601, 2013, 3, n=5)]
        westStations = [s[0] for s in self.wd.stationList(94611, 2013, 3, n=5)]
        eastFlat1 = self.wd.combineStacks(self.wd.weatherMonth(12601, 2013, 3, hourly=False, n=5, stackData=True), addValues=[('zip5', 12601)])
        westFlat1 = self.wd.combineStacks(self.wd.weatherMonth(94611, 2013, 3, hourly=False, n=5, stackData=True), addValues=[('zip5', 94611)])
        eastWestStack = self.wd.weatherMonth([12601, 94611], 2013, 3, hourly=False, n=5, stackData=True)
        eastFlat2 = self.wd.combineStacks(eastWestStack, wbans=eastStations, addValues=[('zip5', 12601)])
        westFlat2 = self.wd.combineStacks(eastWestStack, wbans=westStations, addValues=[('zip5', 94611)])
        assert_frame_equal(eastFlat1, eastFlat2)
        assert_frame_equal(westFlat1, westFlat2)

        marchDataN = self.wd.weatherMonth(12601, 2013, 3, hourly=True, n=5)
        self.assertEqual(len(marchDataN), 12427)

        marchDataDist = self.wd.weatherMonth(12601, 2013, 3, hourly=True, n=0, preferredDistKm=40)
        self.assertEqual(len(marchDataDist), 11483)

    def test_combinedWeatherMonth(self):
        marchDataPA = self.wd.weatherMonth(94301, 2013, 3, hourly=True, n=1)  # PA airport
        combo = self.wd.combineHourlyWeatherData(marchDataPA, removeBlanks=False)
        self.assertTrue(combo['Tmean'].isnull().any())
        combo = self.wd.combineHourlyWeatherData(marchDataPA, removeBlanks=True)
        self.assertFalse(combo['Tmean'].isnull().any())

        marchDataPA = self.wd.weatherMonth(94301, 2013, 3, hourly=True, n=3)  # PA airport
        combo = self.wd.combineHourlyWeatherData(marchDataPA, removeBlanks=False)
        self.assertFalse(combo['Tmean'].isnull().any())  # no blanks anyway

    def test_flattenedWeatherMonths(self):
        start = datetime.datetime(2013, 3, 1)
        end = datetime.datetime(2013, 4, 21)
        flat = self.wd.flattenedWeatherMonths(94301, start, end, hourly=True, preferredDistKm=20)
        self.assertFalse(flat['Tmean'].isnull().any())  # we fixed the blanks!

if __name__ == '__main__':
    unittest.main()
