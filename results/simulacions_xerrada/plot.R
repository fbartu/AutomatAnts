library(ggplot2)
library(latex2exp)

dir <- "G:/research/AutomatAnts/results/simulacions_xerrada/"
fls <- list.files(dir)

L <- vector('list', length(fls))
for(i in seq_along(L)){
        L[[i]] <- read.csv(paste0(dir, fls[i]))[, -1]
        r <-  as.numeric(paste(substr(fls[i], 5, 5),
                                 substr(fls[i], 6, 7), sep = '.'))
        L[[i]]$rho = r
}

lab <- function(string) TeX(paste0("$\\rho = $", string))

L <- do.call('rbind', L)
# 
# ggplot(data = L, aes(Time, Activity)) + geom_line(size = 0.8)+
#         facet_wrap(~ rho,labeller = as_labeller(lab, default = label_parsed)) + 
#         theme_bw() + theme(strip.text = element_text(size = 15, face = 'bold'),
#                            strip.background = element_rect(fill = 'grey90'))
#         
pl <- ggplot(data = L, aes(Time, Activity)) + geom_line(size = 0.6, color = 'navyblue')+
        facet_wrap(~ rho,labeller = as_labeller(lab, default = label_parsed)) + 
        theme_bw() + theme(strip.text = element_text(size = 15, face = 'bold'),
                           strip.background = element_rect(fill = 'grey90'),
                           axis.title = element_text(size = 15),
                           axis.text = element_text(size = 15))+
        scale_y_continuous(breaks = seq(0, 60, 15))

library(svglite)
ggsave(filename = paste0(dir, 'oscilations.svg'), width = 9, height = 6)
ggsave(filename = paste0(dir, 'oscilations.pdf'), width = 9, height = 6)
