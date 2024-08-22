The performance depoly on real robot please checkout link below：

【感觉比之前好多了呢（rl运控实验成功系列）欢迎大家给我的repo加star～-哔哩哔哩】 https://b23.tv/LyjHWJG

打扰大家一下，再给我的repo骗骗star，视频里展示的方案是类似himloco的单阶段训练方案，一样用到了对比学习，不过与himloco的swav不一样的是我用了barlow twin这个算法，它是没有prototype的概念的，同时我的对比学习目标不是拉近history和未来t+1的latent的相似度而是拉近t：t-5 与 t+1:t-4之间的差异（公用mlp encoder与对比学习论文的结构设计更是吻合），我的方案2000轮能收敛到terrain level 6，目前看着还行，据我所知应该没人有这么做过的，由于我是业余爱好，所以我也没发论文的时间，大家有兴趣的话可以看看我的代码一起讨论进步下
