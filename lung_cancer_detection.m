function robt310_finalproject_lungcancer (input_image)

    try
        I = imread(input_image);
    catch ME
        fprintf('Error while reading the image: %s\n', ME.message);
        return;
    end  
    
    % convert if input is rgb
    if size(I, 3) == 3
        I = rgb2gray(I);
    else
        I = I;
    end
    %% Otsu's Thresholding
    
    [m, n] = size(I);
    h = imhist(I);
    p = h / (m * n); % normalized histogram
    
    %the between-class variance (that we have two classes)
    %one= lower than threshold(T), one higher
    alpha_b = 0;
    
    T = 0; % var to store the optimal threshold number
    
    % check all possible threshold values
    for t = 1:255
        
        omega_0 = sum(p(1:t));
        omega_1 = sum(p(t+1:end));
        
        if omega_0 == 0 || omega_1 == 0
            continue % skip if one of the classes=empty
        end
        
        mu_0 = sum(p(1:t) .* (1:t)') / omega_0;  
        mu_1 = sum(p(t+1:end) .* (t+1:256)') / omega_1;  
        
        % the between-class variance according to formula
        alpha_b_t = omega_0 * omega_1 * (mu_1 - mu_0).^2;
        
        % if the variance is larger, it is gonna become a new threshold
        if alpha_b_t > alpha_b
            alpha_b = alpha_b_t;
            T = t;
        end
    end
    
    % the optimal threshold value
    disp(['The optimal threshold is ' num2str(T)])
    
    % threshold the input using T(the optimal threshold) we found
    g = I > T;
    
    imwrite(g, "otsu_image.jpg");
    % compare the original and thresholded images
    figure
    subplot(1,2,1)
    imshow(I)
    title('Original image')
    subplot(1,2,2)
    imshow(g)
    title('Thresholded image')
    
    %% Image Segmentation
    
    I = im2bw(I); 
    
    % Erosion
		     
    se = ones(9, 9); % create structuring element	 
    [P, Q]=size(se); 
     
    In=zeros(size(I, 1), size(I, 2)); 
    
    for i=ceil(P/2):size(I, 1)-floor(P/2)
	    for j=ceil(Q/2):size(I, 2)-floor(Q/2)
    
		    % take all the neighbourhoods.
		    on=I(i-floor(P/2):i+floor(P/2), j-floor(Q/2):j+floor(Q/2)); 
    
		    % take logical se
		    nh=on(logical(se)); 
	    
		    % compare and take minimum value of the neighbor 
		    % and set the pixel value to that minimum value. 
		    In(i, j)=min(nh(:));	 
	    end
    end
    
    % Median Filter
    filterSize = 3;
    filteredImage = medfilt2(In, [filterSize, filterSize]);
    
    figure
    subplot(1,2,1)
    imshow(I)
    title('Thresholded image')
    subplot(1,2,2)
    imshow(filteredImage)
    title('Filtered image')
    
    imwrite(filteredImage, 'filtered.jpg');
    %% Opening
    binaryImage = filteredImage;
    SE = strel('disk', 5); 
    openedImage = imopen(binaryImage, SE);
    
    figure;
    subplot(1, 2, 1);
    imshow(binaryImage);
    title('Original Image');
    axis off;
    
    subplot(1, 2, 2);
    imshow(openedImage);
    title('Image after Opening Operation');
    axis off;
    
    imwrite(openedImage, 'opened_img.jpg');
    %% Area Detection
    binaryImage = openedImage;
    labeledImage = bwlabel(binaryImage);
    
    % measure region properties
    stats = regionprops(labeledImage, 'Area', 'PixelIdxList');
    
    % find the region with the smallest area
    areas = [stats.Area];
    [smallestArea, smallestIdx] = min(areas);
    
    % extract the smallest region's boundary
    boundary = bwboundaries(labeledImage == smallestIdx);
    
    figure;
    subplot(1, 2, 1);
    imshow(binaryImage);
    title('Original Image');
    axis off;
    
    subplot(1, 2, 2);
    imshow(binaryImage);
    hold on;
    plot(boundary{1}(:, 2), boundary{1}(:, 1), 'r', 'LineWidth', 2);
    title('Contour of Smallest Region');
    axis off;
    hold off;
    
    segmentedImage = false(size(binaryImage));
    
    % draw the contour of the smallest region on the new image
    segmentedImage(sub2ind(size(binaryImage), boundary{1}(:, 1), boundary{1}(:, 2))) = true;
    filledRegion = roipoly(binaryImage, boundary{1}(:, 2), boundary{1}(:, 1));
    
    % combine the contour and filled region
    segmentedImage(filledRegion) = true;
    
    figure;
    imshow(segmentedImage);
    title('Segmented Region with Contour and Filled Area');
    axis off;
    
    imwrite(segmentedImage, 'segmented.jpg'); 
    
    %% Feature Extraction 
    
    binaryImage = imread('segmented.jpg');
    info = imfinfo('segmented.jpg'); 
    widthInMillimeters = info.Width; % Width of the image in millimeters
    heightInMillimeters = info.Height; % Height of the image in millimeters
    
    pixelSizeX = widthInMillimeters / size(binaryImage, 2); % Pixel size along the X-axis
    pixelSizeY = heightInMillimeters / size(binaryImage, 1); % Pixel size along the Y-axis
    
    averagePixelSize = (pixelSizeX + pixelSizeY) / 2 % Average pixel size
    
    %%
    
    % Calculate Area and Eccentricity for the Largest Connected Component
    labeledImage = logical(binaryImage);
    stats = regionprops(labeledImage, 'Area', 'PixelIdxList');
    
    % Find the region with the largest area (assuming there is only one)
    [~, largestIdx] = max([stats.Area]);
    
    % Extract properties for the largest connected component
    largestArea = stats(largestIdx).Area;
    largestPixelIdxList = stats(largestIdx).PixelIdxList;
    
    % Calculate Perimeter
    perimeter = bwperim(binaryImage, 8); % Boundary perimeter
    perimeterLength = sum(perimeter(:)) * averagePixelSize;
    
    % Calculate Eccentricity based on the largest connected component
    eccentricity = sqrt(1 - 4 * pi * largestArea / perimeterLength^2);
    
    % Calculate Entropy using GLCM
    glcm = graycomatrix(binaryImage, 'Offset', [0 1], 'NumLevels', 2);
    p = glcm / sum(glcm(:)); % Normalize GLCM
    entropyValue = -sum(p(p > 0) .* log2(p(p > 0))); % Entropy calculation
    
    % Calculate contrast
    offsets = [0 1; -1 1; -1 0; -1 -1];
    num_gray_levels = 256; 
    glcm2 = graycomatrix(binaryImage, 'Offset', offsets, 'NumLevels', num_gray_levels);
    
    properties = graycoprops(glcm2, {'contrast', 'correlation', 'energy', 'homogeneity'});
    
    glcm_norm = glcm2 ./sum(glcm2(:));
    contrast = 0;
    for i = 1:num_gray_levels
        for j = 1:num_gray_levels
            contrast = contrast + ((i - j)^2) * glcm_norm(i, j);
        end
    end
    max_possible_contrast = (num_gray_levels - 1)^2;
    normalized_contrast = contrast / max_possible_contrast;
    
    % Calculate correlation 
    
    correlation_value = properties.Correlation;
    str1 = num2str(correlation_value);
    numbers1 = str2double(strsplit(str1));
    correlation = mean(numbers1);
    
    % Calculate energy
    
    energy_value = properties.Energy;
    str2 = num2str(energy_value);
    numbers2 = str2double(strsplit(str2));
    energy = mean(numbers2);
    
    % Calculate homogeneity
    
    homogeneity_value = properties.Homogeneity;
    str3 = num2str(homogeneity_value);
    numbers3 = str2double(strsplit(str3));
    homogeneity = mean(numbers3);
    
    %% Display values
    
    disp(['Area: ' num2str(largestArea) ' mm^2']);
    disp(['Perimeter: ' num2str(perimeterLength) ' mm']);
    disp(['Eccentricity: ' num2str(eccentricity)]);
    disp(['Entropy: ' num2str(entropyValue)]);
    disp(['Contrast: ', num2str(normalized_contrast)]);
    disp(['Correlation: ', num2str(correlation) ]);
    disp(['Energy: ', num2str(energy)]);
    disp(['Homogeneity: ', num2str(homogeneity)]);
end