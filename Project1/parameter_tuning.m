similarity = 0:0.01:1;
r = 25;
b = 39;

probability = @(s,r, b) 1-(1-s.^r).^b

plot(similarity, probability(similarity,r,b), 'r')
title('Performance Plot')
xlabel('Jaccard Similarity s between two videos')
ylabel('Probability of being hashed to the same bucket')
legend(sprintf(' 1 - (1 - s^r)^b for r = %d and b = %d', r,b), 'location', 'northwest')