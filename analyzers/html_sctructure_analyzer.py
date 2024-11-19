from .base_analyzer import BaseAnalyzer
import requests
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Any


class HTMLStructureAnalyzer(BaseAnalyzer):
    def analyze(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Analyzes HTML content and extracts key elements."""
        self.logger.info("Starting HTML structure analysis")

        elements = self.extract_elements(soup)
        issues = self.check_issues(soup, elements["links"], url)
        recommendations = self.generate_recommendations({**elements, **issues}, soup, url)

        return {
            "elements": elements,
            "issues": issues,
            "recommendations": recommendations,
        }

    def extract_elements(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extracts key elements from the page."""
        return {
            "titles": self.extract_titles(soup),
            "meta": self.extract_meta(soup),
            "links": self.extract_links(soup),
            "images": self.extract_images(soup),
            "buttons": self.extract_buttons(soup),
        }

    def check_issues(self, soup: BeautifulSoup, links: List[Dict[str, str]], url: str) -> Dict[str, Any]:
        """Performs various checks and returns detected issues."""
        return {
            "accessibility_issues": self.check_accessibility(soup),
            "links_status": self.check_links(links, url),
            "deprecated_elements": self.check_deprecated_elements(soup),
            "heading_structure": self.analyze_heading_structure(soup),
            "script_and_css_loading": self.analyze_script_and_css_loading(soup),
        }

    def extract_titles(self, soup: BeautifulSoup) -> List[str]:
        """Extracts titles (h1-h6 and <title>)."""
        return [
            title.get_text(strip=True)
            for title in soup.find_all(["title", "h1", "h2", "h3", "h4", "h5", "h6"])
        ]

    def extract_meta(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extracts meta tags with name or property attributes."""
        meta_info = {}
        for meta in soup.find_all("meta"):
            if meta.get("name") and meta.get("content"):
                meta_info[meta["name"]] = meta["content"]
            elif meta.get("property") and meta.get("content"):
                meta_info[meta["property"]] = meta["content"]
        return meta_info

    def extract_links(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extracts all anchor links."""
        return [
            {"text": link.get_text(strip=True), "href": link["href"]}
            for link in soup.find_all("a", href=True)
        ]

    def extract_images(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extracts image sources and alt attributes."""
        return [
            {"alt": img.get("alt", ""), "src": img["src"]}
            for img in soup.find_all("img", src=True)
        ]

    def extract_buttons(self, soup: BeautifulSoup) -> List[str]:
        """Extracts text of buttons and input[type=button]."""
        buttons = soup.find_all(["button", "input"], {"type": ["button", "submit"]})
        return [button.get_text(strip=True) for button in buttons]

    def check_links(self, links: List[Dict[str, str]], base_url: str) -> List[Dict[str, Any]]:
        """Checks the status of links."""
        statuses = []
        for link in links:
            full_url = self.resolve_url(link["href"], base_url)
            status = self.get_link_status(full_url)
            statuses.append({"url": full_url, "status": status})
        return statuses

    def resolve_url(self, href: str, base_url: str) -> str:
        """Resolves a relative URL to an absolute one."""
        if not re.match(r"^https?://", href):
            return requests.compat.urljoin(base_url, href)
        return href

    def get_link_status(self, url: str) -> str:
        """Performs a HEAD request to check the status of a URL."""
        try:
            response = requests.head(url, timeout=5)
            return response.status_code
        except requests.exceptions.RequestException:
            return "Error"

    def check_accessibility(self, soup: BeautifulSoup) -> List[str]:
        """Checks for accessibility issues."""
        issues = []
        missing_aria = soup.find_all(attrs={"aria-label": False})
        if missing_aria:
            issues.append(f"{len(missing_aria)} elements missing aria-label attributes.")

        non_semantic_tags = soup.find_all(
            ["div", "span"], recursive=True, attrs={"role": None}
        )
        if non_semantic_tags:
            issues.append(f"{len(non_semantic_tags)} non-semantic tags found without roles.")

        return issues

    def check_deprecated_elements(self, soup: BeautifulSoup) -> List[str]:
        """Checks for deprecated tags and attributes."""
        deprecated_tags = {
            "center",
            "font",
            "big",
            "small",
            "strike",
            "u",
            "frame",
            "frameset",
            "noframes",
            "applet",
            "bgsound",
            "basefont",
            "s",
            "tt",
            "i",
            "b",
            "isindex",
            "menu",
            "acronym",
            "dir",
            "listing",
            "plaintext",
            "marquee"
        }
        deprecated_attributes = {"align", "bgcolor"}

        issues = []
        for tag in deprecated_tags:
            if soup.find(tag):
                issues.append(f"Deprecated tag <{tag}> found.")
        for attr in deprecated_attributes:
            if soup.find(attrs={attr: True}):
                issues.append(f"Deprecated attribute '{attr}' found.")
        return issues

    def analyze_heading_structure(self, soup: BeautifulSoup) -> List[str]:
        """Analyzes heading levels for proper structure."""
        headings = [int(h.name[1]) for h in soup.find_all(re.compile("^h[1-6]$"))]
        issues = [
            f"Improper heading structure: <h{headings[i]}> follows <h{headings[i - 1]}>."
            for i in range(1, len(headings))
            if headings[i] > headings[i - 1] + 1
        ]
        return issues

    def analyze_script_and_css_loading(self, soup: BeautifulSoup) -> List[str]:
        """Analyzes JS and CSS loading strategies."""
        issues = []
        scripts = soup.find_all("script", src=True)
        for script in scripts:
            if not script.get("async") and not script.get("defer"):
                issues.append(f"Script {script['src']} is missing 'async' or 'defer'.")
        return issues

    def generate_recommendations(self, data: Dict[str, Any], soup: BeautifulSoup, url: str) -> List[str]:
        """Generates recommendations based on extracted data and issues."""
        recommendations = []

        if not data["titles"]:
            recommendations.append("Add a <title> tag for SEO.")
        if "description" not in data["meta"]:
            recommendations.append("Add a meta description tag for SEO.")

        images_without_alt = [img for img in data["images"] if not img["alt"]]
        if images_without_alt:
            recommendations.append(f"{len(images_without_alt)} images are missing alt attributes.")

        if len(data["links_status"]) > 0:
            broken_links = [link for link in data["links_status"] if link["status"] != 200]
            if broken_links:
                recommendations.append(f"{len(broken_links)} broken links found.")

        return recommendations
