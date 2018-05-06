import unittest
import time
import datetime
import WeatherData as weather
from pandas.util.testing import assert_frame_equal

class Timer(object):
  def __init__(self, name=None):
    self.name = name

  def __enter__(self):
    self.tstart = time.time()

  def __exit__(self, type, value, traceback):
    if self.name:
      print '[%s]' % self.name,
    print 'Elapsed: %s' % (time.time() - self.tstart)

class WeatherTest(unittest.TestCase):

  # can use:
  # self.assertEqual
  # self.assertRaises(TypeError,etc...)
  # self.assertTrue

  def setUp(self):
    self.wd = weather.WeatherData('weather') # dataDir

  def tezt_weatherRange(self):
    zip5 = 12601
    zips  = self.wd.zipMap()
    print(zips[zip5])
    with Timer('weather'):
      start = datetime.datetime(2013,3,1)
      end   = datetime.datetime(2013,4,21)
      dt = datetime.timedelta(days=1)
      dates = [start + x * dt for x in range((end-start).days)]
      #(dates,tout) = wd.matchWeather(dates,zip5,hourly=True)
      start = datetime.datetime(2013,6,1)
      end   = datetime.datetime(2013,6,5)
      #dhr = datetime.timedelta(hours=1)
      #hours = [start + x * dhr for x in range((end-start).days*24)]
      weather = self.wd.weatherRange(zip5,start,end,True)
      print weather
  
  def test_stationList(self):
    
    closest5 = self.wd.stationList(12601,y=2013,m=3,n=5)
    self.assertTrue(len(closest5) == 5)
    print("Top %d stations closest to %d:" % (len(closest5),12601))
    print("  WBAN, dist (km), name")
    for sta in closest5: print("  %s" % self.wd.summarizeStation(sta))
    print('')
    
    within10 = self.wd.stationList(12601,y=2013,m=3,n=0,preferredDistKm=10)
    self.assertTrue(len(within10) == 1)
    print("%d station(s) within %d km from %d:" % (len(within10),10, 12601))
    print("  WBAN, dist (km), name")
    for sta in within10: print("  %s" % self.wd.summarizeStation(sta))
    print('')
    
    within30 = self.wd.stationList(94568,y=2013,m=3,n=0,preferredDistKm=30)
    print("%d station(s) within %d km from %d." % (len(within30),30, 94568))
    print("  WBAN, dist (km), name")
    for sta in within30: print("  %s" % self.wd.summarizeStation(sta))
    self.assertTrue(len(within30) == 3)
    print('')

    broken = self.wd.stationList(95223,y=2013,m=3,n=3)
    print("%d station(s) within %d km from %d." % (len(broken),30, 95223))
    print("  WBAN, dist (km), name")
    for sta in broken: print("  %s" % self.wd.summarizeStation(sta))

  def test_weatherMonth(self):
    with self.assertRaises(KeyError):
      self.wd.weatherMonth(999,2013,3,hourly=False)
    marchDataN = self.wd.weatherMonth(12601,2013,3,hourly=False,n=5)
    #print(len(marchDataN))
    self.assertEqual(len(marchDataN),124)

    marchDataDist = self.wd.weatherMonth(12601,2013,3,hourly=False,n=0,preferredDistKm=40)
    #print(len(marchDataDist))
    self.assertEqual(len(marchDataDist),93)

    # data for a list of zip codes
    marchDataDist = self.wd.weatherMonth([12601,94611],2013,3,hourly=False,n=0,preferredDistKm=40)
    self.assertEqual(len(marchDataDist),279)
    
    # Two ways to get flattened data from stack of source data
    flat1 = self.wd.combineStacks(self.wd.stackDailyWeatherData(marchDataN),addValues=[('zip5',12601)])
    #print flat1
    flat2 = self.wd.combineStacks(self.wd.weatherMonth(12601,2013,3,hourly=False,n=5,stackData=True),addValues=[('zip5',12601)])
    #print flat2
    assert_frame_equal(flat1,flat2)

    if True:
      # calling weatherMonth with stacked=True and a list of zip codes stacks wbans for several zips at once
      # this section of code tests that the outcome of combining those superset stacks with wban subsets
      # gives the same results as flattening monthly data for individual zips
      # the multi-zip approach can prevent multiple reads per zip code of the huge data files.
      eastStations  = [s[0] for s in self.wd.stationList(12601,2013,3,n=5)]
      westStations  = [s[0] for s in self.wd.stationList(94611,2013,3,n=5)]
      eastFlat1     = self.wd.combineStacks(self.wd.weatherMonth(12601,2013,3,hourly=False,n=5,stackData=True),addValues=[('zip5',12601)])
      westFlat1     = self.wd.combineStacks(self.wd.weatherMonth(94611,2013,3,hourly=False,n=5,stackData=True),addValues=[('zip5',94611)])
      eastWestStack = self.wd.weatherMonth([12601,94611],2013,3,hourly=False,n=5,stackData=True)
      eastFlat2     = self.wd.combineStacks(eastWestStack,wbans=eastStations,addValues=[('zip5',12601)])
      westFlat2     = self.wd.combineStacks(eastWestStack,wbans=westStations,addValues=[('zip5',94611)])
      assert_frame_equal(eastFlat1,eastFlat2)
      assert_frame_equal(westFlat1,westFlat2)

    if True:
      marchDataN = self.wd.weatherMonth(12601,2013,3,hourly=True,n=5)
      self.assertEqual(len(marchDataN),12427)

      marchDataDist = self.wd.weatherMonth(12601,2013,3,hourly=True,n=0,preferredDistKm=40)
      self.assertEqual(len(marchDataDist),11483)


  def tezt_combinedWeatherMonth(self):
    marchDataPA = self.wd.weatherMonth(94301,2013,3,hourly=True,n=1) # PA airport
    combo = self.wd.combineHourlyWeatherData(marchDataPA,removeBlanks=False)
    self.assertTrue(combo['Tmean'].isnull().any())
    combo = self.wd.combineHourlyWeatherData(marchDataPA,removeBlanks=True)
    self.assertFalse(combo['Tmean'].isnull().any())

    # fix the blanks with more stations
    marchDataPA = self.wd.weatherMonth(94301,2013,3,hourly=True,n=3) # PA airport
    combo = self.wd.combineHourlyWeatherData(marchDataPA,removeBlanks=False)
    self.assertFalse(combo['Tmean'].isnull().any()) # no blanks anyway
  
  def tezt_flattenedWeatherMonths(self):
    start = datetime.datetime(2013,3,1)
    end   = datetime.datetime(2013,4,21)
    flat = self.wd.flattenedWeatherMonths(94301,start,end,hourly=True,preferredDistKm=20)
    self.assertFalse(flat['Tmean'].isnull().any()) # we fixed the blanks!

if __name__ == '__main__':
  unittest.main()