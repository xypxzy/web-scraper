from .base_analyzer import BaseAnalyzer
import requests

class HTMLStructureAnalyzer(BaseAnalyzer):
    def analyze(self, soup, url):
        """Analyses HTML content and extracts key elements."""
        self.logger.info("Starting HTML structure analysis")

        elements = {
            'titles': self.extract_titles(soup),
            'meta': self.extract_meta(soup),
            'links': self.extract_links(soup),
            'images': self.extract_images(soup),
            'buttons': self.extract_buttons(soup)
        }

        recommendations = self.generate_recommendations(elements, soup, url)
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

    def check_sitemap(self, url):
        """Checks for the existence of a sitemap.xml file."""
        sitemap_url = f"{url.rstrip('/')}/sitemap.xml"
        try:
            response = requests.head(sitemap_url, timeout=5)
            if response.status_code == 200:
                return True, sitemap_url
            return False, sitemap_url
        except requests.exceptions.RequestException:
            return False, sitemap_url

    def analyze_page_speed(self, soup):
        """Basic analysis of page size and optimization hints."""
        total_speed = 0
        for tag in soup.find_all(['img', 'script', 'link']):
            if tag.name == 'img' and tag.get('src'):
                total_speed += self.estimate_resource_size(tag['src'])
            elif tag.name == 'script' and tag.get('src'):
                total_speed += self.estimate_resource_size(tag['src'])
            elif tag.name == 'link' and tag.get('href'):
                total_speed += self.estimate_resource_size(tag['href'])
        return total_speed

    def estimate_resource_size(self, resource_url):
        """Attempts to estimate the size of a resource."""
        try:
            response = requests.head(resource_url, timeout=5)
            if response.status_code == 200 and 'Content-Length' in response.headers:
                return int(response.headers['Content-Length'])
        except requests.exceptions.RequestException:
            pass
        return 0

    def generate_recommendations(self, elements, soup, url):
        """Generates recommendations based on the extracted elements."""
        recommendations = []
        required_tags = ['main', 'header', 'footer']

        # TODO: нужно добавить еще проверок и довести этот метод до конца

        # Check for title tag
        if not elements['titles']:
            recommendations.append("Add a <title> tag to improve SEO.")
        else:
            title_text = elements['titles'][0]
            if len(title_text) < 10 or len(title_text) > 70:
                recommendations.append("The <title> tag should be between 10 and 70 characters long.")

        # Check for meta description tag
        if 'description' not in elements['meta']:
            recommendations.append("Add a ‘description’ meta tag to improve SEO.")
        else:
            description = elements['meta']['description']
            if len(description) < 50 or len(description) > 160:
                recommendations.append("The ‘description’ meta tag should be between 50 and 160 characters long.")

        # Check for images without alt attribute
        images_without_alt = [img for img in elements['images'] if not img['alt']]
        if images_without_alt:
            recommendations.append(f"{len(images_without_alt)} images are missing the 'alt' attribute.")

        # Check for multiple <h1> tags
        h1_tags = [title for title in elements['titles'] if 'h1' in str(title)]
        if len(h1_tags) == 0:
            recommendations.append("Add an <h1> tag to indicate the main title of the page.")
        elif len(h1_tags) > 1:
            recommendations.append("Having more than one <h1> tag can confuse search engines.")

        # Check if required tags are present
        for tag in required_tags:
            if not soup.find(tag):
                recommendations.append(f"Consider adding a <{tag}> tag to improve semantic structure.")

        sitemap_exists, sitemap_url = self.check_sitemap(url)
        if not sitemap_exists:
            recommendations.append(f"No sitemap.xml found. Consider adding one for better SEO. Expected location: {sitemap_url}")
        else:
            recommendations.append(f"Sitemap.xml found at: {sitemap_url}")

        total_size = self.analyze_page_speed(soup)
        if total_size > 2 * 1024 * 1024:  # More than 2 MB
            recommendations.append(f"The page size is large ({total_size / 1024 / 1024:.2f} MB). Consider optimizing resources such as images, scripts, and stylesheets.")

        return recommendations
