---
title: Content Negotiation
description: Aushandlung von Format, Sprache und Kodierung zwischen Client und Server
  über HTTP.
category:
- Architecture
problems:
- poor-interfaces-between-applications
- rest-api-design-issues
- integration-difficulties
- breaking-changes
- legacy-api-versioning-nightmare
layout: solution
lang: de
en_slug: content-negotiation
related_solutions:
- slug: backward-compatible-apis
  similarity: 0.7
- slug: api-versioning-strategy
  similarity: 0.7
- slug: api-gateway
  similarity: 0.7
- slug: standardized-data-formats
  similarity: 0.7
- slug: data-formats
  similarity: 0.65
- slug: api-deprecation-policy
  similarity: 0.65
---

## Description

Content Negotiation erlaubt es einem einzelnen Endpunkt, mehrere Repräsentationen derselben Ressource auszuliefern, indem Client und Server sich über Format, Sprache und Kodierung mittels standardisierter HTTP-Header wie Accept, Accept-Language und Content-Type einigen, statt für jede Repräsentation einen separaten Endpunkt bereitzustellen. Legacy-APIs haben sich häufig früh auf ein einziges Format festgelegt — oft XML —, und als eine neue Klasse von Clients etwas anderes brauchte, etwa JSON für mobile Konsumenten, war der Weg des geringsten Widerstands, parallele, formatspezifische Endpunkte anzuflanschen, die dann auf unbestimmte Zeit synchron zu den ursprünglichen gehalten werden mussten. Content Negotiation vermeidet diese Duplizierung, indem die Formatentscheidung in die Anfrage selbst verlagert wird: Der Server prüft, was der Client zu akzeptieren bereit ist, und antwortet entsprechend, sodass alte und neue Clients während einer schrittweisen Migration zu neuen Formaten über dieselbe Route bedient werden können. Weil es auf etablierten HTTP-Semantiken beruht, ist es für erfahrene API-Konsumenten vorhersehbar und lässt sich natürlich mit API-Versionierungsschemata auf Basis benutzerdefinierter Medientypen kombinieren. Der Mechanismus fügt jedoch Komplexität bei Request-Handling, Serialisierung und Cache-Verhalten hinzu — insbesondere rund um den Vary-Header —, und nicht jeder HTTP-Client implementiert Negotiation korrekt, was eine relevante Einschränkung ist, wenn Legacy-Integrationspartner beteiligt sind.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Implementieren Sie serverseitige Content Negotiation über die standardisierten HTTP-Header Accept, Accept-Language und Content-Type
- Unterstützen Sie mehrere Antwortformate (JSON, XML, CSV) über denselben Endpunkt, statt formatspezifische Endpunkte zu erstellen
- Nutzen Sie Content Negotiation zur API-Versionierung über benutzerdefinierte Medientypen (z. B. application/vnd.company.v2+json)
- Fügen Sie Fallback-Verhalten hinzu, das ein sinnvolles Standardformat zurückgibt, wenn der Client keine Präferenzen angibt
- Dokumentieren Sie unterstützte Medientypen und Negotiation-Verhalten in Ihrer API-Dokumentation
- Testen Sie Content-Negotiation-Pfade als Teil Ihrer Integrationstest-Suite

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Ermöglicht einem einzigen Endpunkt, mehrere Client-Bedürfnisse ohne URL-Wildwuchs zu bedienen
- Unterstützt schrittweise Formatmigration, indem neue Formate neben bestehenden hinzugefügt werden
- Folgt HTTP-Standards, was die API für erfahrene Konsumenten vorhersehbarer macht

**Kosten und Risiken:**
- Fügt Request-Handling und Serialisierungslogik Komplexität hinzu
- Das Debugging kann schwieriger werden, wenn derselbe Endpunkt unterschiedliche Formate zurückgibt
- Nicht alle HTTP-Clients handhaben Content Negotiation korrekt, besonders ältere
- Das Cache-Verhalten wird mit Vary-Headern komplexer

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-Unternehmensanwendung lieferte Daten ausschließlich in XML aus. Als mobile Clients JSON-Antworten benötigten, erstellte das Team zunächst duplizierte Endpunkte. Nach der Implementierung von Content Negotiation trafen sowohl XML- als auch JSON-Konsumenten auf dieselben Endpunkte, wobei der Server das Format basierend auf dem Accept-Header auswählte. Dies eliminierte 30 Prozent der Endpunkt-Duplizierung und erlaubte dem Team später, Protocol-Buffers-Unterstützung für interne Hochdurchsatz-Konsumenten hinzuzufügen, ohne neue Routen anzulegen.
