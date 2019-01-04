%%				  
%%				  
%%	LABORATORY FINAL PROJECT  
%%				  
%%	Mouzakitis Nikolaos TP4460
%%				  
%%				  
%%				  
pkg load statistics;
pkg load nan;

data = load('data');
realVA = [];
realSK = [];
realCU = [];
realEN = [];
forgedVA = [];
forgedSK = [];
forgedCU = [];
forgedEN = [];

cat = zeros(1372,1);

%	Real banknotes.
cat(1:762,1) = 1;
 %	Forged banknotes.
cat(763:1372,1) = 2; 

countR = 0;
countF = 0;

for i = 1 : 1372

	if(data(i,5) == 0)
		countR = countR+1;	
		realVA(countR,1) = data(i,1);
		realSK(countR,1) = data(i,2);
		realCU(countR,1) = data(i,3);
		realEN(countR,1) = data(i,4);	
	else
		countF = countF + 1;
		forgedVA(countF,1) = data(i,1);
		forgedSK(countF,1) = data(i,2);
		forgedCU(countF,1) = data(i,3);
		forgedEN(countF,1) = data(i,4);	
	endif
endfor

%%	Creation of sublots/boxplots to do the part
%%	of statistical analysis and choose features.

figure(1)
subplot(1,2,1)
boxplot(realVA);, title('Real Variance'), axis([0 2 -10 10]);
subplot(1,2,2), 
boxplot(forgedVA);,title('Forged Variance'), axis([0 2 -10 10]);

figure(2)
subplot(1,2,1)
boxplot(realSK);, title('Real Skewness'), axis([0 2 -15 15]);
subplot(1,2,2), 
boxplot(forgedSK);,title('Forged Skewness'), axis([0 2 -15 15]);

figure(3)
subplot(1,2,1)
boxplot(realCU);, title('Real Curtosis'), axis([0 2 -15 19]);
subplot(1,2,2), 
boxplot(forgedCU);,title('Forged Curtosis'), axis([0 2 -15 19]);

figure(4)
subplot(1,2,1)
boxplot(realEN);, title('Real Entropy'), axis([0 2 -15 19]);
subplot(1,2,2), 
boxplot(forgedEN);,title('Forged Entropy'), axis([0 2 -15 19]);

%%	After this step , we are choosing the features 1,2( Variance and Skewness)
%%	to use in the implementation of our classifiers, since with this combination
%%	we can distinguish better than the other possible choices real from forged
%%	banknotes.
%%	Check for missing attributes it is not performed since in the 
A = data(:,1);
B = data(:,2);
%%	Normalization of the values in the closed [0,1].
lmin = min(min(A)); lmax = max(max(A));

for i = 1:1372
	A(i) = (A(i) - lmin) / (lmax-lmin);
endfor

lmin = min(min(B)); lmax = max(max(B));

for i = 1:1372
	B(i) = (B(i) - lmin) / (lmax-lmin);
endfor

C = [A B];

for i = 1:1372
	newMatrix(i,:) = [ C(i,1); C(i,2); cat(i)];
endfor

cat = cat';

%%	80/20 split.    610 real/488 forged.
%%***************************************
TrainingSet = zeros(1098,3);
ValidationSet = zeros(274,3);

%% inserting the real elements in TS.
for i = 1:610
	TrainingSet(i,1) = A(i);
	TrainingSet(i,2) = B(i);
	TrainingSet(i,3) = 1;
endfor

%% get the forged elements as well in the TS
j = 1;
for i = 611: 1098

	TrainingSet(i,1) = A(762+j,1);
	TrainingSet(i,2) = B(762+j,1);
	TrainingSet(i,3) = 2;
	j = j+1;	
endfor

%% Now the same for the Validation Set.

for i = 1:152
	ValidationSet(i,1) = A(610+i,1);
	ValidationSet(i,2) = B(610+i,1);
	ValidationSet(i,3) = 1;	
endfor

for i = 1: 122
	ValidationSet(152+i,1) = A(1250+i,1);
	ValidationSet(152+i,2) = B(1250+i,1);
	ValidationSet(152+i,3) = 2;
endfor

spTrain = zeros(1098,1);
spVal = zeros(274,1);
for i = 1: 610
	spTrain(i) = 1;
endfor

for i = 611: 1098
	spTrain(i) = 2;
endfor

for i = 1: 152
	spVal(i) = 1;
endfor

for i = 153: 274
	spVal(i) = 2;
endfor

%%***************************************
%%		LINEAR CLASSIFIER	 
%%%	Gscatter: quick visualization of the 80/20 training/validation samples.
figure(5), gscatter(newMatrix(:,1), newMatrix(:,2),cat, 'rgb', 'xo+')
title('All samples')
figure(6), gscatter(TrainingSet(:,1), TrainingSet(:,2),spTrain, 'rgb', 'xo+');
title('Training Set');
figure(7), gscatter(ValidationSet(:,1), ValidationSet(:,2),spVal, 'rgb', 'xo+');
title('Validation Set');

ValidationSet(:,3) = [];
TrainingSet(:,3) = [];

[classesL, errnL] = classify(ValidationSet, TrainingSet, spTrain, 'LDA');

disp('80-20 error for Linear classifier by Classify Function');
errnL*100

error82 = 0;

for i = 1:274
	if( classesL(i) == spVal(i))
		error82 = error82+1;
	endif	
endfor

disp('80-20 error for Linear classifier step by step instance on the Validation Set');
100 - (error82/274)*100

[x,y] = meshgrid(0:0.01:1,0:0.01:1);	
x = x(:);
y = y(:);

j1 = classify([x y], TrainingSet(:,[1,2]) , spTrain, 'LDA');
figure(8)
subplot(1,3,1)
gscatter(x,y,j1,'rgb','ooo');
title('Linear Classifier 80-20 SPLIT');
axis([0 1 0 1]);
subplot(1,3,2);
gscatter(ValidationSet(:,1), ValidationSet(:,2),spVal, 'rgb', 'xo+');
title('Pre Classification');
axis([0 1 0 1]);
subplot(1,3,3);		
gscatter(ValidationSet(:,1), ValidationSet(:,2),classesL,'rgb','xo+');
title('Post Classification');
axis([ 0 1 0 1 ] );

%%%	Gscatter: quick visualization of the 70/30 training/validation samples.
%%	70/30 split.    960/412.
%%***************************************

TrainingSet2 = zeros(960,3);
ValidationSet2 = zeros(412,3);

%% inserting the real elements in TS.
for i = 1:480
	TrainingSet2(i,1) = A(i);
	TrainingSet2(i,2) = B(i);
	TrainingSet2(i,3) = 1;
endfor

%% get the forged elements as well in the TS
j = 1;
for i = 481 : 480+480
	TrainingSet2(i,1) = A(762+j,1);
	TrainingSet2(i,2) = B(762+j,1);
	TrainingSet2(i,3) = 2;
	j = j+1;	
endfor

%% Now the same for the Validation Set.

for i = 1:282
	ValidationSet2(i,1) = A(480+i,1);
	ValidationSet2(i,2) = B(480+i,1);
	ValidationSet2(i,3) = 1;	
endfor

for i = 1: 130 
	ValidationSet2(282+i,1) = A(762+480+i,1);
	ValidationSet2(282+i,2) = B(762+480+i,1);
	ValidationSet2(282+i,3) = 2;
endfor

spTrain2 = zeros(960,1);
spVal2 = zeros(412,1);

for i = 1: 480
	spTrain2(i) = 1;
endfor

for i = 481: 960 
	spTrain2(i) = 2;
endfor

for i = 1: 762-480
	spVal2(i) = 1;
endfor

for i = 762-480+1 : 412
	spVal2(i) = 2;
endfor

[classesL2, errnL2] = classify(ValidationSet2(:,1:2), TrainingSet2(:,1:2), spTrain2, 'LDA');
disp('70-30 error for Linear classifier given by Classify function');
errnL2*100


error73 = 0;
for i = 1:412
	if( classesL2(i) == spVal2(i))
		error73 = error73+1;
	endif	
endfor
disp('70-30 error for Linear classifier step by step instance applied on the Validation Set');
100 - (error73/412)*100


[xx,yy] = meshgrid(0:0.01:1,0:0.01:1);	
xx = xx(:);
yy = yy(:);

ValidationSet2(:,3) = [];
TrainingSet2(:,3) = [];

j12 = classify([xx yy], TrainingSet2(:,[1,2]) , spTrain2, 'LDA');
figure(18)
subplot(1,3,1)
gscatter(x,y,j12,'rgb','ooo');
title('Linear Classifier 70-30 SPLIT');
axis([0 1 0 1]);
subplot(1,3,2);
gscatter(ValidationSet2(:,1), ValidationSet2(:,2),spVal2, 'rgb', 'xo+');
title('Pre Classification');
axis([0 1 0 1]);
subplot(1,3,3);		
gscatter(ValidationSet2(:,1), ValidationSet2(:,2),classesL2,'rgb','xo+');
title('Post Classification');
axis([ 0 1 0 1 ] );

%///********************
%%%	Gscatter: quick visualization of the 90/10 training/validation samples.
%%	1235/137.
%%***************************************

TrainingSet2 = zeros(1235,3);
ValidationSet2 = zeros(137,3);

%% inserting the real elements in TS.
for i = 1:694
	TrainingSet2(i,1) = A(i);
	TrainingSet2(i,2) = B(i);
	TrainingSet2(i,3) = 1;
endfor

%% get the forged elements as well in the TS
j = 1;
for i = 695:695+540
	TrainingSet2(i,1) = A(762+j,1);
	TrainingSet2(i,2) = B(762+j,1);
	TrainingSet2(i,3) = 2;
	j = j+1;	
endfor

%% Now the same for the Validation Set.

for i = 1: 68
	ValidationSet2(i,1) = A(694+i,1);
	ValidationSet2(i,2) = B(694+i,1);
	ValidationSet2(i,3) = 1;	
endfor

for i = 1: 69 
	ValidationSet2(68+i,1) = A(762+541+i,1);
	ValidationSet2(68+i,2) = B(762+541+i,1);
	ValidationSet2(68+i,3) = 2;
endfor

spTrain2 = zeros(1235,1);
spVal2 = zeros(137,1);

for i = 1: 694 
	spTrain2(i) = 1;
endfor

for i = 695: 1235 
	spTrain2(i) = 2;
endfor

for i = 1: 68
	spVal2(i) = 1;
endfor

for i = 69 : 137 
	spVal2(i) = 2;
endfor

[classesL2, errnL2] = classify(ValidationSet2(:,1:2), TrainingSet2(:,1:2), spTrain2, 'LDA');
disp('90-10 error for Linear classifier given by Classify function');
errnL2*100

error91 = 0;
for i = 1:137
	if( classesL2(i) == spVal2(i))
		error91 = error91+1;
	endif	
endfor
disp('90-10 error for Linear classifier step by step instance applied on the Validation Set');
100 - (error91/137)*100

[xx,yy] = meshgrid(0:0.01:1,0:0.01:1);	
xx = xx(:);
yy = yy(:);

ValidationSet2(:,3) = [];
TrainingSet2(:,3) = [];

j12 = classify([xx yy], TrainingSet2(:,[1,2]) , spTrain2, 'LDA');
figure(20)
subplot(1,3,1)
gscatter(x,y,j12,'rgb','ooo');
title('Linear Classifier 90-10 SPLIT');
axis([0 1 0 1]);
subplot(1,3,2);
gscatter(ValidationSet2(:,1), ValidationSet2(:,2),spVal2, 'rgb', 'xo+');
title('Pre Classification');
axis([0 1 0 1]);
subplot(1,3,3);		
gscatter(ValidationSet2(:,1), ValidationSet2(:,2),classesL2,'rgb','xo+');
title('Post Classification');
axis([ 0 1 0 1 ] );

%////***
%%	Implementing the Quadratic Classifier.

[classesQ, errnQ] = classify(ValidationSet, TrainingSet, spTrain, 'QDA2');
disp('80-20 error for quadratic classifier given by Classify function: ');
errnQ*100

error82 = 0;
for i = 1:274
	if( classesQ(i) == spVal(i))
		error82 = error82+1;
	endif	
endfor
disp('80-20 error for Quadratic classifier step by step instance applied on the Validation Set');
100 - (error82/274)*100

j2 = classify([x y], TrainingSet(:, [1,2]), spTrain, 'QDA2');
figure(9)
subplot(1,3,1)
gscatter(x,y,j2,'rgb','ooo');
title('Quadratic Classifier 80-20 SPLIT');
axis([0 1 0 1]);
subplot(1,3,2);
gscatter(ValidationSet(:,1), ValidationSet(:,2),spVal, 'rgb', 'xo+');
title('Pre Classification');
axis([0 1 0 1]);
subplot(1,3,3);		
gscatter(ValidationSet(:,1), ValidationSet(:,2),classesQ,'rgb','xo+');
title('Post Classification');
axis([ 0 1 0 1 ] );


%%Quadratic 70/30 split.

%%	70/30 split.    960/412.
%%***************************************

TrainingSet2 = zeros(960,3);
ValidationSet2 = zeros(412,3);

%% inserting the real elements in TS.
for i = 1:480
	TrainingSet2(i,1) = A(i);
	TrainingSet2(i,2) = B(i);
	TrainingSet2(i,3) = 1;
endfor

%% get the forged elements as well in the TS
j = 1;
for i = 481 : 480+480
	TrainingSet2(i,1) = A(762+j,1);
	TrainingSet2(i,2) = B(762+j,1);
	TrainingSet2(i,3) = 2;
	j = j+1;	
endfor

%% Now the same for the Validation Set.

for i = 1:282
	ValidationSet2(i,1) = A(480+i,1);
	ValidationSet2(i,2) = B(480+i,1);
	ValidationSet2(i,3) = 1;	
endfor

for i = 1: 130 
	ValidationSet2(282+i,1) = A(762+480+i,1);
	ValidationSet2(282+i,2) = B(762+480+i,1);
	ValidationSet2(282+i,3) = 2;
endfor

spTrain2 = zeros(960,1);
spVal2 = zeros(412,1);

for i = 1: 480
	spTrain2(i) = 1;
endfor

for i = 481: 960 
	spTrain2(i) = 2;
endfor

for i = 1: 762-480
	spVal2(i) = 1;
endfor

for i = 762-480+1 : 412
	spVal2(i) = 2;
endfor

[classesL2, errnL2] = classify(ValidationSet2(:,1:2), TrainingSet2(:,1:2), spTrain2, 'QDA');
disp('70-30 error for Quadratic classifier');
errnL2*100

error73 = 0;

for i = 1:412
	if( classesL2(i) == spVal2(i))
		error73 = error73 + 1;
	endif	
endfor
disp('70-30 error for Quadratic classifier step by step instance applied on the Validation Set');
100 - (error73/412)*100



[xx,yy] = meshgrid(0:0.01:1,0:0.01:1);	
xx = xx(:);
yy = yy(:);

ValidationSet2(:,3) = [];
TrainingSet2(:,3) = [];

j12 = classify([xx yy], TrainingSet2(:,[1,2]) , spTrain2, 'QDA');
figure(19)
subplot(1,3,1)
gscatter(x,y,j12,'rgb','ooo');
title('Quadratic Classifier 70-30 SPLIT');
axis([0 1 0 1]);
subplot(1,3,2);
gscatter(ValidationSet2(:,1), ValidationSet2(:,2),spVal2, 'rgb', 'xo+');
title('Pre Classification');
axis([0 1 0 1]);
subplot(1,3,3);		
gscatter(ValidationSet2(:,1), ValidationSet2(:,2),classesL2,'rgb','xo+');
title('Post Classification');
axis([ 0 1 0 1 ] );

%%%	Gscatter: quick visualization of the 90/10 training/validation samples for quadratic classifier.
%%	1235/137.
%%***************************************

TrainingSet2 = zeros(1235,3);
ValidationSet2 = zeros(137,3);

%% inserting the real elements in TS.
for i = 1:694
	TrainingSet2(i,1) = A(i);
	TrainingSet2(i,2) = B(i);
	TrainingSet2(i,3) = 1;
endfor

%% get the forged elements as well in the TS
j = 1;
for i = 695:695+540
	TrainingSet2(i,1) = A(762+j,1);
	TrainingSet2(i,2) = B(762+j,1);
	TrainingSet2(i,3) = 2;
	j = j+1;	
endfor

%% Now the same for the Validation Set.

for i = 1: 68
	ValidationSet2(i,1) = A(694+i,1);
	ValidationSet2(i,2) = B(694+i,1);
	ValidationSet2(i,3) = 1;	
endfor

for i = 1: 69 
	ValidationSet2(68+i,1) = A(762+541+i,1);
	ValidationSet2(68+i,2) = B(762+541+i,1);
	ValidationSet2(68+i,3) = 2;
endfor

spTrain2 = zeros(1235,1);
spVal2 = zeros(137,1);

for i = 1: 694 
	spTrain2(i) = 1;
endfor

for i = 695: 1235 
	spTrain2(i) = 2;
endfor

for i = 1: 68
	spVal2(i) = 1;
endfor

for i = 69 : 137 
	spVal2(i) = 2;
endfor

[classesL2, errnL2] = classify(ValidationSet2(:,1:2), TrainingSet2(:,1:2), spTrain2, 'QDA');
disp('90-10 error for Quadratic classifier as given by Classify function:');
errnL2*100

error91 = 0;
for i = 1:137
	if( classesL2(i) == spVal2(i))
		error91 = error91+1;
	endif	
endfor
disp('90-10 error for Quadratic classifier step by step instance applied on the Validation Set');
100-(error91/137)*100

[xx,yy] = meshgrid(0:0.01:1,0:0.01:1);	
xx = xx(:);
yy = yy(:);

ValidationSet2(:,3) = [];
TrainingSet2(:,3) = [];

j12 = classify([xx yy], TrainingSet2(:,[1,2]) , spTrain2, 'QDA');
figure(20)
subplot(1,3,1)
gscatter(x,y,j12,'rgb','ooo');
title('Quadratic Classifier 90-10 SPLIT');
axis([0 1 0 1]);
subplot(1,3,2);
gscatter(ValidationSet2(:,1), ValidationSet2(:,2),spVal2, 'rgb', 'xo+');
title('Pre Classification');
axis([0 1 0 1]);
subplot(1,3,3);		
gscatter(ValidationSet2(:,1), ValidationSet2(:,2),classesL2,'rgb','xo+');
title('Post Classification');
axis([ 0 1 0 1 ] );



%%	k-means clustering.

data = [A B];		


%%	Best distance method after trying various others choosen to be the 'sqeuclidean'.
[idx, centers] = kmeans(data, 2,'DISTANCE', 'sqeuclidean');
figure(10)
plot( data(idx==1,1),data(idx ==1,2),'r*');
hold on;
plot( data(idx==2,1),data(idx ==2,2),'g+');
title('K means classification');
hold off;

%% to evaluate the k-means :: find most conquering values and	       
%% set correctly one and two's in the idx in order to compare with cat.

cc = 0;

for i = 1:762
	if idx(i,1) == 1
		cc = cc + 1;
	endif
endfor

%% in case there is actually a problem
if ( cc < (762 / 2) )

	for i=1:1372

		if(idx(i,1)== 1)
			idx(i,1) = 2;
		else
			idx(i,1) = 1;
		endif
	endfor
endif

%%calculation of error for the k-mean clustering.
errK = 0;

for i = 1: 1372
	if(cat'(i,1) != idx(i,1))
		errK = errK+1;
	endif
endfor

disp('Error of k-means')
errK/1372 * 100

%% Hierarchical clustering.
load data;
data = data(:,1:4);

D = pdist(data);
y = linkage(D);

figure(11)
dendrogram(y);
disp('Inc = ');
inc = inconsistent(y)
