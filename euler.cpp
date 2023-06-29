void euler(std::vector<std::pair<std::vector<float>, int>> &result, float threshold) {
    for (int x = 0; x < result.size(); x++) {
        result[x].first = tuple(result[x].first);
        for (int y = x; y < result.size(); y++) {
            if (sqrt(sum(square(array(result[x].first) - array(result[y].first)))) < threshold) {
                result[y].first = tuple(result[x].first);
            }
            else {
                result[y].first = tuple(result[y].first);
            }
        }
    }
    return result.drop_duplicates(subset=['percentage']).size, result.value_counts().tolist() / sum(result.value_counts().tolist());
}
