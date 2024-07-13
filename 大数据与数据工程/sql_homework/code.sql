-- Active: 1711636777825@@127.0.0.1@3306
CREATE DATABASE IF NOT EXISTS movie_Management
    DEFAULT CHARACTER SET = 'utf8mb4'; 
USE movie_Management;
# 建立用户表，id作为主键
CREATE TABLE `user`(
  `id` VARCHAR(20) NOT NULL,
  `name` VARCHAR(20) NOT NULL,
  `password` VARCHAR(20) NOT NULL,
  PRIMARY KEY (`id`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
# 建立video表
CREATE TABLE `video`(
  `id` VARCHAR(20) NOT NULL,
  `name` VARCHAR(20) NOT NULL,
  `type` VARCHAR(20) NOT NULL,
  `score` DECIMAL(10,1) NOT NULL,
  `starring` VARCHAR(200) NOT NULL,
  `date` DATE NOT NULL,
  `description` TEXT NOT NULL,
  `path` VARCHAR(200) NOT NULL,
  PRIMARY KEY (`id`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
# 建立comment表
CREATE TABLE `comment`(
  `id` VARCHAR(20) NOT NULL,
  `user_id` VARCHAR(20) NOT NULL,
  `description` TEXT NOT NULL,
  `date` DATE NOT NULL,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`user_id`) REFERENCES `user`(`id`) 
)ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
# 建立favorite表
CREATE TABLE `favorite`(
  `user_id` VARCHAR(20) NOT NULL,
  `video_id` VARCHAR(20) NOT NULL,
  PRIMARY KEY (`user_id`, `video_id`),
  FOREIGN KEY (`user_id`) REFERENCES `user`(`id`),
  FOREIGN KEY (`video_id`) REFERENCES `video`(`id`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 插入用户数据
INSERT INTO `user` (`id`, `name`, `password`) VALUES
('u1', 'Jack', 'pw1'),
('u2', 'Nancy', 'pw2'),
('u3', 'George', 'pw3'),
('u4', 'Amy', 'pw4'),
('u5', 'Martin', 'pw5'),
('u6','Adam','pw6'),
('u7', 'uzi', 'pw7'),
('u8','jiejie','pw8');

SELECT * FROM user;
-- 插入影视数据
INSERT INTO `video` (`id`, `name`, `type`, `score`, `starring`, `date`, `description`, `path`) VALUES
('v1', '我和我的祖国', '剧情', 8.2, '黄渤', '2019-09-30', '一部讲述中国历史和人民生活的影片.','路径1'),
('v2', '火星救援', '科幻', 8.0, '马特·达蒙', '2015-11-25', '一部描绘人类逆境求生的电影.','路径2'),
('v3', '你的名字', '动画', 8.4, '神木隆之介', '2016-12-02', '一部描绘时间穿越、命运交织的故事.','路径3'),
('v4', '那些年，我们一起追的女孩', '爱情', 7.4, '柯震东', '2011-08-19', '一部描绘青春的爱情故事.','路径4'),
('v5', '波西米亚狂想曲', '传记', 8.2, '拉米·马雷克', '2019-03-22', '一部讲述摇滚乐团Queen主唱的人物传记.', '路径5'),
('v6', '致命魔术', '剧情', 3.5, '詹森·斯坦森', '2006-10-20', '一部被批评为过于夸张的剧情电影.','路径6'),
('v7', '搏击之王', '动作', 2.8, '斯蒂芬·席格尔', '2002-06-06', '一部质感清淡、情节无新意的动作片.','路径7'),
('v8', '大卫·贝肯之霸王别姬', '喜剧', 1.6, '大卫·贝肯', '2012-07-27', '一部低俗，毫无逻辑的低端喜剧.','路径8'),
('v9', '野蛮人巴巴', '奇幻', 2.3, '施瓦辛格', '1982-02-25', '一部被批评为太过商业化的奇幻电影.','路径9'),
('v10', '母女大战', '家庭', 3.2, '琳赛·洛翰', '2006-05-01', '一部被批评为尴尬，令人难以忍受的家庭电影.','路径10');

SELECT * FROM video;

DROP TABLE user;
DROP TABLE comment;
DROP TABLE video;
DROP TABLE favorite;

-- 插入留言数据
INSERT INTO comment (`id`, `user_id`, `description`, `date`) VALUES
('c1', 'u1', '2151406喜欢看动作电影.', '2024-03-30'),
('c2', 'u2', '无敌破坏王是我最喜欢的动画电影之一，很有创意.', '2024-03-29'),
('c3', 'u2', '星际穿越的剧情有点复杂，但还是喜欢它的科幻元素.', '2024-03-27'),
('c4', 'u3', '2151406喜欢看动作电影.', '2024-03-25'),
('c5', 'u2', '2151406喜欢看动作电影.', '2024-03-22'),
('c6', 'u1', '致命魔术实在是让人疑惑，难以理解其剧情.', '2024-03-20'),
('c7', 'u3', '搏击之王的动作场面需要再加工，特效也很一般.', '2024-03-18'),
('c8', 'u4', '大卫·贝肯之霸王别姬实在太低俗，理解不了为什么会有人喜欢.', '2024-03-15'),
('c9', 'u4', '期待野蛮人巴巴能更注重故事的构造, 而不是只看重商业效益.', '2024-03-12'),
('c10', 'u5', '母女大战的剧情有些尴尬, 对演员本身的浪费.', '2024-03-11'),
('c11', 'u6', '2151406喜欢看动作电影.', '2024-03-11'),
('12', 'u7', '有操作的呀.', '2024-03-15'),
('c13', 'u7', '烂的一匹，洗澡去了.', '2024-03-12'),
('c14', 'u7', '2151406喜欢看动作电影.', '2024-03-11'),
('c15', 'u8', '2151406喜欢看动作电影.', '2024-03-11');

SELECT * FROM comment;
-- 插入收藏数据
INSERT INTO `favorite` (`user_id`, `video_id`) VALUES
('u1', 'v1'),
('u1', 'v3'),
('u1', 'v4'),
('u1', 'v10'),
('u2', 'v1'),
('u2', 'v4'),
('u2', 'v5'),
('u2', 'v7'),
('u2', 'v8'),
('u3', 'v4'),
('u3', 'v2'),
('u3', 'v1'),
('u4', 'v5'),
('u3', 'v3'),
('u5', 'v6'),
('u5', 'v5'),
('u6', 'v2'),
('u6', 'v6'),
('u6', 'v9'),
('u7', 'v6'),
('u7', 'v8'),
('u7', 'v5'),
('u7', 'v9'),
('u7', 'v10'),
('u8', 'v1'),
('u8', 'v2');

SELECT * FROM favorite;
-- 问题b
--方法1，采用join函数的方法，把四个表直接连接起来进行查找
SELECT user.name as b
FROM user
JOIN comment ON user.id = comment.user_id
JOIN favorite ON user.id = favorite.user_id
JOIN video on favorite.video_id = video.id
WHERE comment.description = '2151406喜欢看动作电影.'
GROUP BY user.id
HAVING AVG(video.`score`) > 6.0;

--方法2，
SELECT name
FROM user
WHERE id IN (
  SELECT user_id
  FROM comment
  WHERE description = '2151406喜欢看动作电影.'
) AND id IN (
  SELECT user_id
  FROM favorite
  INNER JOIN video ON video_id = video.id
  GROUP BY user_id
  HAVING AVG(score) > 6
);

-- 方法3
SELECT name
FROM user
WHERE id IN (
  SELECT user_id
  FROM comment
  WHERE description = '2151406喜欢看动作电影.'
)
AND id IN (
  SELECT user_id
  FROM (
    SELECT user_id, AVG(score) AS avg_score
    FROM favorite
    INNER JOIN video ON video_id = video.id
    GROUP BY user_id
  ) AS avg_scores
  WHERE avg_score > 6
);

-- 方法4
CREATE TEMPORARY TABLE TempUsers
SELECT name
FROM user u
WHERE EXISTS (
  SELECT 1
  FROM comment c
  WHERE c.user_id = u.id AND description = '2151406喜欢看动作电影.'
)
AND EXISTS (
  SELECT 1
  FROM favorite f
  JOIN video v ON f.video_id = v.id
  WHERE f.user_id = u.id
  HAVING AVG(v.score) > 6
);

SELECT * FROM tempusers;
DROP TABLE tempusers;
-- 清空所选用户的留言内容
UPDATE comment 
SET description = ''
WHERE user_id IN (
    SELECT u.id
    FROM user u
    INNER JOIN TempUsers t ON u.name = t.name
);

SELECT * FROM comment;
--情况所选用户的整条留言
DELETE comment 
FROM comment
JOIN user ON comment.user_id = user.id
WHERE user.name IN (
    SELECT name
    FROM TempUsers
);
SELECT * FROM comment;
 