library("GGally")

weather <- read.csv("daily-weather-dataset_chronological-order.csv")
solar <- read.csv("solar-data-daily.csv")
solar <- subset(solar, select = -c(New.Nexus.1272.Meter, Site.Performance.Estimate))
colnames(solar)[1] <- "Date"

all <- merge(solar, weather, by = "Date")
all <- all[complete.cases(all), ]

all$Inverters <- as.numeric(levels(all$Inverters))[all$Inverters]
all$Cloud.coverage <- as.numeric(levels(all$Cloud.coverage))[all$Cloud.coverage]
all <- subset(all, select = -c(X.Inverters.))



load("all.corr.Rda")
all.corr$variable <- mapvalues(all.corr$variable, 
                               from=c("alt" , "wind" , "hum" ,  "dew"  , "cloud", "vis"), 
                               to=c("Altimeter","Wind speed",
                                    "Relative humidity", "Dew point", "Cloud coverage", "Visibility"))

all.corr$Month <- mapvalues(all.corr$Month, 
                            from=c("1" , "2" , "3" ,  "4"  , "5", "6", "7", "8", "9", "10", "11", "12"),
                            to = c("January", "February", "March", "April", "May", "June", "July",
                                   "August", "September", "October", "November", "December"))
      
all.corr <- all.corr %>%
  mutate(type = ifelse((COR > 0.5 | COR < -0.5), "h1", "l"))               

ggplot(data = all.corr, aes(x = Month, y = COR, fill = type)) +
  geom_bar(stat="identity") + facet_wrap(~variable) + theme_bw() +
  ylab("Correlation") + scale_fill_manual(values=c("firebrick2", "grey80", "firebrick2"))+
  guides(fill=FALSE)

ggsave("corr-v1.pdf", width = 8, height = 5)
