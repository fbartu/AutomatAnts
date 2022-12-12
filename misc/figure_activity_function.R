## PLOTS XERRADA FEDE
library(ggplot2)
library(reshape2)
library(RColorBrewer)
library(latex2exp)
x <- seq(-10, 10, 0.1)
y <- tanh(x)
g <- seq(0, 1, 0.25)
data <- data.frame(x = x, y = y)
for(i in seq_along(g)){
     name <- paste0('y', i)
     data[[name]] <- tanh(x * g[i])
}
data[['y']] <- NULL

df <- melt(data,id.vars = c('x'))
 
pl <- ggplot(data = df, aes(x, value, color = variable)) + geom_line(size = 2.5) +
     scale_x_continuous(TeX('Input activity, $\\sum J_{\\textit{ij}}S_{\\textit{j}}(\\textit{t})$'), 
                        breaks = seq(-10, 10, 2.5)) +
     ylab(TeX('Output activity, $S_{\\textit{i}}(\\textit{t}+1)$'))+ theme_bw() +
     scale_color_brewer('Gain (g)', palette = 'Dark2', 
                        labels = format(seq(0, 1, 0.25), nsmall = 2))+
     theme(axis.text = element_text(size = 33, color = 'black'),
           axis.title = element_text(size = 33, color = 'black'),
           axis.ticks = element_line(color = 'black', size = 2.5),
           legend.text = element_text(size = 33),
           legend.title = element_text(size = 33))+
     annotate('label', x = -5, y = 0.75, size = 12,
              label = TeX('$S_{\\textit{i}}(\\textit{t}+1) = \\Theta [g(\\sum J_{\\textit{ij}}S_{\\textit{j}}(\\textit{t})-\\theta_{\\textit{i}})]$'))+
     annotate('label', x = -5.55, y = 0.5, size = 12,
              label = TeX('$S_{\\textit{i}}(\\textit{t}+1) = tanh [g(\\sum S_{\\textit{j}}(\\textit{t}))]$'))

pl


png('~/Desktop/figure_activity_function.png', 1920, 1080, res = 100)
pl
dev.off()
