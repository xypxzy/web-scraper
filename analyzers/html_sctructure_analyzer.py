from .base_analyzer import BaseAnalyzer


class HTMLStructureAnalyzer(BaseAnalyzer):
    def analyze(self, soup):
        """Analyses HTML content and extracts key elements."""
        self.logger.info("Starting HTML structure analysis")

        elements = {
            'titles': self.extract_titles(soup),
            'meta': self.extract_meta(soup),
            'links': self.extract_links(soup),
            'images': self.extract_images(soup),
            'buttons': self.extract_buttons(soup)
        }

        recommendations = self.generate_recommendations(elements)
        return {'elements': elements, 'recommendations': recommendations}

    def extract_titles(self, soup):
        """Extracts the title of the page."""
        titles = soup.find_all(["title", "h1", "h2", "h3", "h4", "h5", "h6"])
        return [title.get_text(strip=True) for title in titles]

    def extract_meta(self, soup):
        """Extracts the meta tags of the page."""
        meta_tags = soup.find_all("meta")
        meta_info = {}
        for meta in meta_tags:
            if "name" in meta.attrs and "content" in meta.attrs:
                meta_info[meta["name"]] = meta["content"]
            elif "property" in meta.attrs and "content" in meta.attrs:
                meta_info[meta["property"]] = meta["content"]
        return meta_info

    def extract_links(self, soup):
        """Extracts the links in the page."""
        links = soup.find_all("a", href=True)
        return [{"text": link.get_text(strip=True), 'href': link['href']} for link in links]

    def extract_images(self, soup):
        """Extracts the images in the page."""
        images = soup.find_all("img", src=True)
        return [{'alt': img.get('alt', ''), 'src': img['src']} for img in images]

    def extract_buttons(self, soup):
        """Extracts the buttons, input['button'] in the page."""
        buttons = soup.find_all(['button', 'input'], {'type': ['button', 'submit']})
        return [button.get_text(strip=True) for button in buttons]

    def generate_recommendations(self, elements):
        """Generates recommendations based on the extracted elements."""
        recommendations = []

        # TODO: нужно добавить еще проверок и довести этот метод до конца
        if not elements['titles']:
            recommendations.append("Add a <title> tag to improve SEO.")
        else:
            title_text = elements['titles'][0]
            if len(title_text) < 10 or len(title_text) > 70:
                recommendations.append("The <title> tag should be between 10 and 70 characters long.")

        if 'description' not in elements['meta']:
            recommendations.append("Add a ‘description’ meta tag to improve SEO.")
        else:
            description = elements['meta']['description']
            if len(description) < 50 or len(description) > 160:
                recommendations.append("The ‘description’ meta tag should be between 50 and 160 characters long.")

        images_without_alt = [img for img in elements['images'] if not img['alt']]
        if images_without_alt:
            recommendations.append(f"{len(images_without_alt)} images are missing the 'alt' attribute.")

        h1_tags = [title for title in elements['titles'] if 'h1' in str(title)]
        if len(h1_tags) == 0:
            recommendations.append("Add an <h1> tag to indicate the main title of the page.")
        elif len(h1_tags) > 1:
            recommendations.append("Having more than one <h1> tag can confuse search engines.")

        return recommendations
