# Title     : TODO
# Objective : TODO
# Created by: gaetan
# Created on: 29/05/17

require(plyr)
require(ggplot2)
require(reshape2)
require(grid)

df = read.csv('/home/gaetan/Projets/Python/workspace/constraints/log.csv', header = FALSE)
names(df) <- c(c('epoch_index', 'mean_val_loss', 'mean_val_accuracy', 'constraint_val_accuracy'), llply(0:10, function(x){paste0('pi', x)}))

pi = df[grepl('pi', names(df))]

pi_melted = melt(as.matrix(pi))


df$t = df$epoch_index / 100
df_melted = melt(df, id.vars = 't')
df_melted = subset(df_melted, grepl('val', variable))

g1 = ggplot(pi_melted, aes(x =Var1, y = Var2)) + geom_tile(aes(fill=value )) +
  theme(axis.title.x = element_blank(), axis.text.x = element_blank()) +
  scale_fill_gradient2(low = 'blue', high='red', mid='lightblue', midpoint=0.095)
   #scale_fill_gradientn(colours = rainbow(8))


df_loss = subset(df_melted, variable == "mean_val_loss")
g2 = ggplot(df_loss, aes(x = t, y=value, colour=variable)) + geom_point() + theme(axis.title.x = element_blank(), axis.text.x = element_blank())

df_accuracy = subset(df_melted, grepl('accuracy', variable))
g3 = ggplot(df_accuracy, aes(x = t, y=value, colour=variable)) + geom_point() + 
  theme(axis.title.x = element_blank(), axis.text.x = element_blank()) +
  ylim(0, 1)


grid.newpage()
grid.draw(rbind(ggplotGrob(g1), ggplotGrob(g2), ggplotGrob(g3), size = "last"))