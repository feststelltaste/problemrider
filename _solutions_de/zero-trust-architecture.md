---
title: Zero Trust Architecture
description: „Niemals vertrauen, immer verifizieren“ — Überprüfung jeder
  Anfrage unabhängig vom Netzwerkstandort.
category:
- Security
- Architecture
problems:
- authentication-bypass-vulnerabilities
- authorization-flaws
- monolithic-architecture-constraints
- system-integration-blindness
- configuration-drift
- poor-interfaces-between-applications
- insecure-data-transmission
layout: solution
lang: de
en_slug: zero-trust-architecture
related_solutions:
- slug: trust-boundaries
  similarity: 0.75
- slug: security-architecture-analysis
  similarity: 0.7
- slug: web-application-firewall
  similarity: 0.7
- slug: security-by-design
  similarity: 0.7
- slug: network-segmentation
  similarity: 0.7
- slug: security-certification
  similarity: 0.7
---

## Description

Zero Trust Architecture ist ein Sicherheitsmodell, gebaut auf dem Prinzip "niemals vertrauen, immer verifizieren": Keiner Anfrage wird Zugang gewährt, nur weil sie von innerhalb eines bestimmten Netzwerksegments stammt, und jeder Zugriffsversuch wird stattdessen authentifiziert, autorisiert und gegen Kontext bewertet — Identität, Gerätezustand und angefragte Ressource —, unabhängig davon, woher er kommt. Dies ersetzt das traditionelle perimeterbasierte Modell, in dem alles innerhalb der Firewall implizit vertraut wurde, durch kontinuierliche, Pro-Anfrage-Verifikation, durchgesetzt durch Mechanismen wie Identity-Aware Proxies, mutual TLS und feingranulare Zugriffsrichtlinien, angewendet auf der Ebene einzelner Dienste statt Netzwerkzonen. In Legacy-Umgebungen ist diese Verschiebung besonders folgenreich, weil solche Systeme häufig unter der entgegengesetzten Annahme gestaltet wurden: interner Traffic wurde standardmäßig vertraut, Authentifizierung geschah einmal am Netzwerkrand, und Komponenten kommunizierten miteinander mit wenig Rücksicht darauf, wer oder was tatsächlich aufrief. Genau diese Annahme ist es, die einer einzigen kompromittierten Zugangsdaten oder einem verletzten Host erlaubt, sich in uneingeschränkte laterale Bewegung über einen ganzen Bestand miteinander verbundener Legacy-Anwendungen zu verwandeln. Die Nachrüstung von Zero-Trust-Prinzipien auf ein solches System — typischerweise durch das Platzieren von Identity-Aware Proxies vor Legacy-Anwendungen, die nicht nativ jede Anfrage authentifizieren können, und durch Mikrosegmentierung des Netzwerks, um laterale Bewegung einzudämmen — reduziert den Blast-Radius einer Verletzung, ohne notwendigerweise zu erfordern, dass die Legacy-Anwendungen selbst neu geschrieben werden. Der Kompromiss ist, dass diese Nachrüstung architektonisch invasiv und selten schnell ist: Sie berührt Netzwerktopologie, Authentifizierungsflüsse und Inter-Service-Kommunikation gleichzeitig, und wird realistischerweise als langfristige, inkrementelle Modernisierungsanstrengung verfolgt, statt als diskretes Projekt mit einem definierten Enddatum.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Beseitigen Sie implizites Vertrauen basierend auf Netzwerkstandort, indem Sie Authentifizierung und Autorisierung für jede Anfrage verlangen
- Implementieren Sie identitätsbasierte Zugriffskontrollen, die Nutzer, Gerät und Kontext für jeden Zugriffsversuch verifizieren
- Führen Sie Mikrosegmentierung ein, um laterale Bewegung zwischen Legacy-Systemkomponenten einzuschränken
- Setzen Sie einen Identity-Aware Proxy oder ein API-Gateway vor Legacy-Anwendungen ein, denen native Zero-Trust-Fähigkeiten fehlen
- Verschlüsseln Sie alle Kommunikationskanäle, einschließlich internen Traffics zwischen Legacy-Komponenten
- Implementieren Sie kontinuierliches Monitoring und Protokollierung aller Zugriffsentscheidungen zur Anomalieerkennung
- Wenden Sie Prinzipien der geringsten Rechte auf die gesamte Service-zu-Service-Kommunikation in der Legacy-Umgebung an

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Reduziert die Auswirkung von Netzwerkverletzungen dramatisch, indem implizite Vertrauenszonen beseitigt werden
- Bietet granulare Zugriffskontrolle, die sich an Kontext anpasst, statt sich auf statische Netzwerkgrenzen zu verlassen
- Verbessert die Sicherheitssichtbarkeit durch umfassende Zugriffsprotokollierung und -überwachung
- Unterstützt moderne Hybrid- und Cloud-Deployment-Modelle für die Legacy-Systemmigration

**Kosten und Risiken:**
- Die Nachrüstung von Zero Trust in Legacy-Systemen, die vertrauenswürdige Netzwerke annehmen, erfordert erhebliche architektonische Änderungen
- Performance-Overhead durch die Verifikation jeder Anfrage kann latenzsensible Legacy-Anwendungen beeinträchtigen
- Die betriebliche Komplexität steigt erheblich mit Pro-Anfrage-Authentifizierung und -Autorisierung
- Legacy-Protokolle und -Integrationen könnten die Identitäts- und Verschlüsselungsanforderungen von Zero Trust nicht unterstützen
- Vollständige Zero-Trust-Implementierung ist eine mehrjährige Reise, kein einzelnes Projekt

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein großes Unternehmen begann seine Zero-Trust-Reise nach einer Sicherheitsverletzung, bei der ein Angreifer eine kompromittierte VPN-Verbindung nutzte, um sich frei über ihr internes Netzwerk zu bewegen und auf Legacy-Systeme zuzugreifen, die standardmäßig allem internen Traffic vertrauten. Sie begannen, indem sie einen Identity-Aware Proxy vor ihren kritischsten Legacy-Anwendungen einsetzten, der Pro-Anfrage-Authentifizierung auch von internen Nutzern verlangte. Sie fügten dann mutual TLS zwischen den Legacy-Anwendungsservern und der Datenbankschicht hinzu. Innerhalb eines Jahres war das interne Netzwerk in Zonen mit expliziten Zugriffsrichtlinien segmentiert. Eine nachfolgende Red-Team-Übung bestätigte, dass die Kompromittierung eines einzelnen internen Hosts keinen Zugang zu anderen Systemen mehr bot, ein starker Kontrast zur Zero-Trust-Vorzustands-Lage.
