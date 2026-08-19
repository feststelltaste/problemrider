---
title: Prädiktives Laden
description: Proaktives Laden von Daten, die wahrscheinlich als Nächstes
  benötigt werden.
category:
- Performance
problems:
- slow-application-performance
- slow-response-times-for-lists
- high-api-latency
- poor-user-experience-ux-design
- network-latency
- user-frustration
layout: solution
lang: de
en_slug: predictive-loading
related_solutions:
- slug: predictive-prefetching
  similarity: 0.85
- slug: progressive-loading
  similarity: 0.8
- slug: lazy-loading
  similarity: 0.8
- slug: performance-optimization
  similarity: 0.8
- slug: lazy-evaluation
  similarity: 0.8
- slug: optimistic-ui-updates
  similarity: 0.75
---

## Description

Prädiktives Laden antizipiert, welche Daten oder Ressourcen ein Nutzer wahrscheinlich als Nächstes benötigt, basierend auf beobachteten Navigationsmustern, und ruft sie proaktiv während der Leerlaufzeit nach Abschluss der aktuellen Aktion ab, statt darauf zu warten, dass der Nutzer sie explizit anfordert. Die Vorhersage kann so einfach sein wie eine feste, aus der Nutzungsprotokollanalyse abgeleitete Heuristik — die meisten Nutzer, die einen Fall öffnen, sehen sofort dessen Anhänge —, aber in jedem Fall ist das Ziel, ein sequenzielles Warten in Arbeit umzuwandeln, die sich mit der eigenen Denk- oder Lesezeit des Nutzers überschneidet. Dies ist speziell für Legacy-Systeme wertvoll, weil es einem langsamen, monolithischen Backend, dessen Neuarchitektur teuer oder riskant wäre, erlaubt, sich für den Endnutzer erheblich schneller anzufühlen, ohne die zugrunde liegende Datenzugriffsschicht oder Datenbankperformance überhaupt anzufassen — die wahrgenommene Latenz sinkt, obwohl sich die tatsächliche Latenz pro Anfrage nicht ändert. Da es vollständig von der Vorhersagegenauigkeit abhängt, muss prädiktives Laden mit Überwachung der Trefferquoten und elegantem Fallback-Verhalten für die Fälle gepaart werden, in denen die Vermutung falsch ist, da eine falsche Vorhersage Serverressourcen und Bandbreite verschwendet, statt einen funktionalen Fehler zu verursachen. Es wirft auch Fragen der Datenfrische und des Datenschutzes auf: Vorhergesagter Inhalt muss mit sinnvollem Ablauf zwischengespeichert werden, damit er nicht veraltet, bevor er angezeigt wird, und der Aufbau eines Nutzungsmustermodells aus Nutzerverhaltensdaten führt zu denselben Datenschutzverpflichtungen wie jede andere Verhaltensverfolgung.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Analysieren Sie Nutzer-Navigationsmuster und Nutzungsdaten, um zu identifizieren, welche Ressourcen nach einer gegebenen Aktion am wahrscheinlichsten benötigt werden
- Laden Sie Daten für die wahrscheinlichste nächste Nutzeraktion während der Leerlaufzeit nach Abschluss der aktuellen Aktion vor
- Implementieren Sie prädiktives Laden auf API-Ebene, indem verwandte Daten in Antworten eingeschlossen werden, wenn die Kosten gering sind
- Verwenden Sie Browser-Hinweise (rel="preload", rel="prefetch") für statische Assets auf Seiten, die Nutzer wahrscheinlich als Nächstes besuchen
- Cachen Sie vorhergesagte Daten mit angemessenen TTLs, damit veraltete Daten keine Inkonsistenzen verursachen
- Überwachen Sie die Vorhersagegenauigkeit und passen Sie Ladestrategien basierend auf tatsächlichen Nutzungsmustern an
- Implementieren Sie elegante Fallbacks, sodass die Anwendung auch dann korrekt funktioniert, wenn Vorhersagen falsch sind

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Reduziert die wahrgenommene Latenz für häufige Nutzer-Workflows erheblich
- Lässt Legacy-Anwendungen reaktionsschneller wirken, ohne Backend-Optimierung
- Nutzt Leerlauf-Netzwerk- und CPU-Zeit, die sonst verschwendet würde
- Kann schrittweise für die häufigsten Nutzerpfade implementiert werden

**Kosten und Risiken:**
- Falsche Vorhersagen verschwenden Bandbreite und Serverressourcen beim Laden ungenutzter Daten
- Erhöht den gesamten Ressourcenverbrauch, was für ressourcenbeschränkte Legacy-Systeme problematisch sein kann
- Veraltete vorhergesagte Daten können Verwirrung verursachen, wenn sie angezeigt werden, bevor die tatsächliche Anfrage abgeschlossen ist
- Fügt der Caching- und Datenverwaltungsschicht Komplexität hinzu
- Datenschutzbedenken, wenn Nutzerverhaltensmuster für Vorhersagezwecke verfolgt und gespeichert werden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Fallverwaltungssystem erforderte, dass Support-Mitarbeiter durch eine Fallliste navigierten, einzelne Fälle öffneten und dann auf verwandte Dokumente zugriffen. Jeder Übergang beinhaltete ein vollständiges Seitenladen vom Server, im Durchschnitt 3 Sekunden pro Navigation. Durch die Analyse von Nutzungsprotokollen fand das Team heraus, dass 85 Prozent der Mitarbeiter den neuesten Fall in ihrer Warteschlange öffneten und sofort auf dessen Anhänge zugriffen. Sie implementierten prädiktives Laden, das die Details und Anhänge des obersten Falls abrief, sobald die Fallliste geladen war. Für die Mehrheit der Mitarbeiter erschien die Fall-Detailseite sofort, was den Arbeitsablauf von einer frustrierenden Abfolge von Wartezeiten in eine flüssige Erfahrung verwandelte.
